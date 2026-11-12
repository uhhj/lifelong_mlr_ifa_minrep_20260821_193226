const fs=require('fs'),path=require('path'),crypto=require('crypto');
const sharp=require('C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1104-2876-0_26.1104.2876.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules/sharp');
async function main(){
 const src=path.join(__dirname,'DEBR_B_refined.svg');
 const png=path.join(__dirname,'DEBR_B_refined.png');
 const preview=await sharp(src).png().toBuffer();
 fs.writeFileSync(path.join(__dirname,'DEBR_B_preview.png'),preview);
 await sharp(src,{density:192}).withMetadata({density:192}).png().toFile(png);
 const pngPreview=await sharp(png).resize(1800,940).png().toBuffer();
 const labels=Buffer.from('<svg width="3660" height="100" xmlns="http://www.w3.org/2000/svg"><rect width="3660" height="100" fill="white"/><text x="900" y="46" font-family="Times New Roman" font-size="32" text-anchor="middle" fill="#193247">Editable SVG (rendered)</text><text x="2760" y="46" font-family="Times New Roman" font-size="32" text-anchor="middle" fill="#193247">Refined PNG (4800 × 2507)</text><text x="1830" y="86" font-family="Times New Roman" font-size="23" text-anchor="middle" fill="#536A7B">One vector master · Identical layout, wording and colors</text></svg>');
 await sharp({create:{width:3660,height:1040,channels:4,background:'white'}}).composite([{input:preview,left:0,top:0},{input:pngPreview,left:1860,top:0},{input:labels,left:0,top:940}]).png().toFile(path.join(__dirname,'DEBR_B_SVG_PNG_comparison.png'));
 const metadata=await sharp(png).metadata();
 const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
 const receipt={source:'DEBR_B_refined.svg',source_sha256:sha(src),png_sha256:sha(png),width:metadata.width,height:metadata.height,mode:'single SVG source rendered with Sharp; no independent redraw'};
 fs.writeFileSync(path.join(__dirname,'RENDER_RECEIPT.json'),JSON.stringify(receipt,null,2)+'\n');
 console.log(JSON.stringify(receipt));
}
main().catch(e=>{console.error(e);process.exit(1)});
