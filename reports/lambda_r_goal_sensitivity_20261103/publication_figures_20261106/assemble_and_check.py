"""Assemble the six figures and render every final PDF for visual inspection."""
from pathlib import Path
import json
import hashlib
from pypdf import PdfReader, PdfWriter
import pypdfium2 as pdfium
from PIL import Image, ImageOps, ImageDraw

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures'
REVIEW=ROOT/'review'
REVIEW.mkdir(exist_ok=True)
names=['fig01_success_curves','fig02_task_heatmaps','fig03_gate_tradeoffs','fig04_lambda_sensitivity','figS01_task_differences','figS02_lambda_fwt_nbt']
writer=PdfWriter();results=[];tiles=[]
for name in names:
    p=OUT/(name+'.pdf');reader=PdfReader(p)
    assert len(reader.pages)==1
    page=reader.pages[0]
    fonts=[]
    for font in page['/Resources']['/Font'].get_object().values():
        f=font.get_object();assert f['/Subtype']!='/Type3'
        members=f['/DescendantFonts'] if f['/Subtype']=='/Type0' else [f]
        for desc in members:
            desc=desc.get_object();fd=desc['/FontDescriptor'].get_object()
            assert any(k in fd for k in ['/FontFile','/FontFile2','/FontFile3'])
        fonts.append(str(f['/BaseFont']))
    assert fonts and ('OBJECT' in page.extract_text()) and ('GOAL' in page.extract_text())
    png=Image.open(OUT/(name+'.png'))
    width,height=float(page.mediabox.width),float(page.mediabox.height)
    assert abs(png.width-width/72*600)<2 and abs(png.height-height/72*600)<2
    doc=pdfium.PdfDocument(str(p));render=doc[0].render(scale=2.5).to_pil().convert('RGB')
    render.save(REVIEW/(name+'_pdf.png'))
    tile=Image.new('RGB',(900,700),'white');small=ImageOps.contain(render,(880,650));tile.paste(small,((900-small.width)//2,35+(650-small.height)//2))
    ImageDraw.Draw(tile).text((20,10),name,fill='#17212B');tiles.append(tile)
    results.append(dict(file=name+'.pdf',pages=1,fonts_embedded=True,no_type3=True,fonts=fonts,width_pt=width,height_pt=height,png_pixels=[png.width,png.height],dpi=600,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    writer.append(reader)
writer.add_metadata({'/Title':'OBJECT and GOAL: publication figures','/Author':'MLR+IFA experiment analysis'})
book=OUT/'all_figures.pdf'
with book.open('wb') as f:writer.write(f)
assert len(PdfReader(book).pages)==6
sheet=Image.new('RGB',(1800,2100),'#E8EDF0')
for i,tile in enumerate(tiles):sheet.paste(tile,((i%2)*900,(i//2)*700))
sheet.save(REVIEW/'contact_sheet.png')
(ROOT/'EXPORT_AUDIT.json').write_text(json.dumps(dict(status='PASS_PDF_FONTS_DIMENSIONS_AND_RENDERING',figures=results,book_pages=6,book_sha256=hashlib.sha256(book.read_bytes()).hexdigest()),indent=2)+'\n')
manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(OUT.iterdir()) if p.is_file()}
(ROOT/'FIGURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(status='PASS_PDF_FONTS_DIMENSIONS_AND_RENDERING',figures=len(results),book_pages=6),indent=2))
