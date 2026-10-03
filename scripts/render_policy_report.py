"""Build offline HTML and a print PDF from the mathematical specification."""
import asyncio
import json
import os
from pathlib import Path

import markdown
from playwright.async_api import async_playwright

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/math'

async def main():
    source=(REPORT/'POLICY_MATHEMATICS.md').read_text()
    md=markdown.Markdown(extensions=['tables','fenced_code','toc','pymdownx.arithmatex'],extension_configs={'toc':{'toc_depth':'2-3'},'pymdownx.arithmatex':{'generic':True}})
    body=md.convert(source)
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Grounded CARLA policy: mathematics and training guide</title>
<link rel="stylesheet" href="assets/katex/katex.min.css">
<style>
:root{font-family:Georgia,serif;color:#192a3b;background:#edf1f5}body{margin:0;line-height:1.65}main{max-width:1040px;padding:50px 64px;margin:24px auto;background:white;box-shadow:0 8px 32px #172b4214}h1,h2,h3,th,nav{font-family:system-ui,sans-serif;line-height:1.3}h1{font-size:34px;color:#103648;margin-bottom:20px}h2{font-size:25px;margin-top:48px;border-top:2px solid #dce7ed;padding-top:18px;color:#103648}h3{font-size:19px;margin-top:32px}p,li{font-size:16px}a{color:#0e6b87}nav{background:#f1f6f8;padding:20px 24px;border-left:4px solid #2e8891;margin:30px 0}nav ul{list-style:none;padding-left:18px}nav li{font-size:14px;margin:6px 0}nav>div>ul{padding-left:0}nav a{text-decoration:none}table{width:100%;border-collapse:collapse;font-size:13px;margin:24px 0}th{background:#e7f0f4;text-align:left}td,th{padding:10px;border:1px solid #ccdade;vertical-align:top}tr:nth-child(even){background:#f6f9fa}code{font-family:monospace;font-size:.9em;background:#f0f4f6;padding:1px 4px}pre{white-space:pre-wrap;font-size:13px;background:#f0f4f6;padding:18px}blockquote{border-left:4px solid #58a2b4;padding-left:18px}.arithmatex{overflow-x:auto}.katex{font-size:1.03em}.katex-display{padding:8px 0;margin:12px 0}.mermaid{margin:24px 0;background:#f5f9fa;border:1px solid #d7e3e8;padding:15px;text-align:center}.mermaid svg{max-width:100%;height:auto}button{font:14px system-ui;border:0;border-radius:5px;padding:10px 16px;background:#176679;color:white;cursor:pointer}.toolbar{font:14px system-ui;color:#476375;display:flex;gap:20px;align-items:center;margin-bottom:25px}
@media(max-width:700px){main{padding:25px 20px;margin:0}h1{font-size:27px}table{display:block;overflow:auto}}
@page{size:A4;margin:16mm 14mm 18mm}
@media print{:root{background:white}body{line-height:1.45}main{max-width:none;padding:0;margin:0;box-shadow:none}p,li{font-size:10.5pt}h1{font-size:23pt}h2{font-size:16pt;margin-top:24px;break-after:avoid}h3{font-size:12pt;break-after:avoid}nav{break-after:page}nav li{font-size:9pt}table{font-size:8pt}td,th{padding:6px}tr,.katex-display{break-inside:avoid}.mermaid{break-inside:avoid;padding:6px}.mermaid svg{max-height:650px}.toolbar{display:none}pre{font-size:8pt}.katex{font-size:1em}a{text-decoration:none}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
</style><script src="assets/katex/katex.min.js"></script><script src="assets/mermaid/mermaid.min.js"></script></head><body><main>
<div class="toolbar"><button onclick="window.print()">Print / save PDF</button><a href="POLICY_MATHEMATICS.pdf">Download PDF</a><a href="POLICY_MATHEMATICS.md">Markdown source</a></div>
{{BODY}}
</main><script>
window.renderFailures=[];
for (const element of document.querySelectorAll('.arithmatex')){
 const text=element.textContent;
 const display=text.startsWith('\\\\[');
 const expression=text.slice(2,-2);
 try{katex.render(expression,element,{displayMode:display,throwOnError:true,strict:'ignore'});}catch(error){window.renderFailures.push(String(error));}
}
for(const code of document.querySelectorAll('code.language-mermaid')){
 const element=document.createElement('div');element.className='mermaid';element.textContent=code.textContent;code.parentElement.replaceWith(element);
}
mermaid.initialize({startOnLoad:false,theme:'neutral',securityLevel:'strict',flowchart:{htmlLabels:false,useMaxWidth:true,rankSpacing:24,nodeSpacing:30}});
window.reportReady=mermaid.run({querySelector:'.mermaid'}).then(()=>document.fonts.ready).then(()=>{window.renderComplete=true;}).catch(error=>{window.renderFailures.push(String(error));window.renderComplete=true;});
</script></body></html>'''
    # Place a browsable contents list after the first report paragraph block.
    first_h2=body.index('<h2')
    body=body[:first_h2]+'<nav aria-label="Contents"><strong>Contents</strong>'+md.toc+'</nav>'+body[first_h2:]
    page=page.replace('{{BODY}}',body)
    target=REPORT/'POLICY_MATHEMATICS.html'
    target.write_text(page)
    # Use the temporary browser downloaded for this workspace when available;
    # otherwise let Playwright use its standard installation location.
    if Path('/tmp/grounded-report-browser').exists():
        os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH','/tmp/grounded-report-browser')
    async with async_playwright() as browser_tools:
        browser=await browser_tools.chromium.launch(args=['--no-sandbox'])
        tab=await browser.new_page(viewport={'width':1280,'height':1000})
        errors=[]
        tab.on('pageerror',lambda error:errors.append(str(error)))
        await tab.goto(target.as_uri(),wait_until='load')
        await tab.wait_for_function('window.renderComplete === true',timeout=30000)
        failures=await tab.evaluate('window.renderFailures')
        overflow=await tab.evaluate("Array.from(document.querySelectorAll('.katex-display')).filter(x=>x.scrollWidth>x.clientWidth+2).map(x=>x.textContent.slice(0,100))")
        checks={'equations':await tab.locator('.katex').count(),'diagrams':await tab.locator('.mermaid svg').count(),'render_failures':failures,'page_errors':errors,'wide_display_equations':overflow}
        (REPORT/'render_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        if failures or errors:
            raise RuntimeError(checks)
        await tab.screenshot(path=str(REPORT/'report_preview.png'))
        await tab.pdf(path=str(REPORT/'POLICY_MATHEMATICS.pdf'),format='A4',print_background=True,display_header_footer=True,header_template='<span></span>',footer_template='<div style="font-family:Arial;font-size:8px;width:100%;text-align:center;color:#567">Grounded CARLA expert policy · <span class="pageNumber"></span> / <span class="totalPages"></span></div>',margin={'top':'16mm','bottom':'18mm','left':'14mm','right':'14mm'})
        await browser.close()
    print(json.dumps(checks,indent=2))
    print('Report:',target)

if __name__=='__main__':
    asyncio.run(main())
