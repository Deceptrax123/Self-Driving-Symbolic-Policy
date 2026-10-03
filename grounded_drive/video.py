"""Record genuine synchronized CARLA RGB cameras with policy annotations."""
from __future__ import annotations
from pathlib import Path
from queue import Queue, Empty
import textwrap
import time
import numpy as np


class RolloutVideo:
    def __init__(self, world, ego, path, fps=20):
        import carla
        try:
            import imageio_ffmpeg
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as exc:
            raise RuntimeError("Install video dependencies: uv pip install --python .venv/bin/python -e '.[video]'") from exc
        self.Image, self.ImageDraw = Image, ImageDraw
        font_path='/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
        self.font=ImageFont.truetype(font_path,16) if Path(font_path).exists() else ImageFont.load_default(size=16)
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.sensors=[]
        self.queues=[]
        self.frames=0
        self.writer=None
        self.closed=False
        try:
            blueprint=world.get_blueprint_library().find('sensor.camera.rgb')
            blueprint.set_attribute('image_size_x','640')
            blueprint.set_attribute('image_size_y','360')
            blueprint.set_attribute('fov','90')
            blueprint.set_attribute('sensor_tick','0.0')
            poses=[carla.Transform(carla.Location(x=-8,z=5),carla.Rotation(pitch=-18)),
                   carla.Transform(carla.Location(x=1.4,z=1.6))]
            for pose in poses:
                sensor=world.spawn_actor(blueprint,pose,attach_to=ego)
                self.sensors.append(sensor)
                queue=Queue()
                self.queues.append(queue)
                sensor.listen(queue.put)
            self.writer=imageio_ffmpeg.write_frames(str(self.path),(1280,480),fps=fps,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',quality=8,macro_block_size=16,output_params=['-movflags','+faststart'])
            self.writer.send(None)
        except Exception:
            self.close()
            raise

    @staticmethod
    def matching_image(queue, frame):
        deadline=time.monotonic()+15
        while True:
            try:
                image=queue.get(timeout=max(.01,deadline-time.monotonic()))
            except Empty as exc:
                raise RuntimeError(f'No RGB image for CARLA frame {frame}; video requires rendering enabled') from exc
            if image.frame==frame:
                return image
            if image.frame>frame or time.monotonic()>deadline:
                raise RuntimeError(f'RGB frame mismatch: expected {frame}, received {image.frame}')

    def record(self, frame, t, observation, decision, scenario):
        canvas=self.Image.new('RGB',(1280,480),(13,20,29))
        for index,queue in enumerate(self.queues):
            camera=self.matching_image(queue,frame)
            bgra=np.frombuffer(camera.raw_data,dtype=np.uint8).reshape(camera.height,camera.width,4)
            rgb=np.ascontiguousarray(bgra[:,:,:3][:,:,::-1])
            canvas.paste(self.Image.fromarray(rgb),(index*640,0))
        draw=self.ImageDraw.Draw(canvas)
        for x,label in [(0,'CHASE CAMERA'),(640,'FRONT CAMERA')]:
            draw.rectangle((x,0,x+640,28),fill=(13,20,29))
            draw.text((x+12,4),label,font=self.font,fill=(239,244,250))
        control=decision.control
        draw.text((12,367),f'{scenario}   t={t:.2f}s   speed={observation.speed*3.6:.1f} km/h   subgoal={decision.plan.subgoal}',font=self.font,fill=(117,223,193))
        draw.text((12,391),f'throttle={control.throttle:.2f} brake={control.brake:.2f} steer={control.steer:.2f}  surprise={decision.surprisal:.2f}  model updates={decision.model_updates}',font=self.font,fill=(232,237,246))
        facts=', '.join(f.predicate for f in decision.plan.facts) or 'clear route'
        draw.text((12,415),'Grounded facts: '+facts[:118],font=self.font,fill=(232,237,246))
        reason=textwrap.shorten(decision.plan.reason,width=120,placeholder='...')
        draw.text((12,439),reason,font=self.font,fill=(190,205,224))
        self.writer.send(canvas.tobytes())
        self.frames+=1
        if self.frames==1:
            canvas.save(self.path.with_name('preview.png'))

    def close(self, destroy_sensors=True):
        if self.closed:
            return
        self.closed=True
        try:
            for sensor in self.sensors:
                if destroy_sensors and sensor.is_alive:
                    sensor.stop()
                    sensor.destroy()
        finally:
            if self.writer is not None:
                self.writer.close()
            if self.frames==0:
                self.path.unlink(missing_ok=True)
