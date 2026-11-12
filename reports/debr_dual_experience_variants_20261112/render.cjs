const fs=require('fs');
const path=require('path');
const sharp=require('C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1104-2876-0_26.1104.2876.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules/sharp');
async function main(){
 for(const v of ['A','B','C']){
  const svg=path.join(__dirname,`DEBR_${v}.svg`);
  if(!fs.existsSync(svg))continue;
  await sharp(svg,{density:144}).png().toFile(path.join(__dirname,`DEBR_${v}_refined.png`));
  const preview=await sharp(svg).png().toBuffer();
  fs.writeFileSync(path.join(__dirname,`DEBR_${v}_preview.png`),preview);
  const rasterPreview=await sharp(path.join(__dirname,`DEBR_${v}_refined.png`)).resize(1800,940).png().toBuffer();
  const labels=Buffer.from(`<svg width="3660" height="100" xmlns="http://www.w3.org/2000/svg"><rect width="3660" height="100" fill="white"/><text x="900" y="45" font-family="Times New Roman" font-size="32" text-anchor="middle" fill="#193247">${v} · Editable SVG (rendered)</text><text x="2760" y="45" font-family="Times New Roman" font-size="32" text-anchor="middle" fill="#193247">${v} · Refined PNG (2× export)</text><text x="1830" y="85" font-family="Times New Roman" font-size="23" text-anchor="middle" fill="#536A7B">Same vector master · Same composition · Same typography</text></svg>`);
  await sharp({create:{width:3660,height:1040,channels:4,background:'#FFFFFF'}}).composite([{input:preview,left:0,top:0},{input:rasterPreview,left:1860,top:0},{input:labels,left:0,top:940}]).png().toFile(path.join(__dirname,`DEBR_${v}_comparison.png`));
  console.log(`Rendered ${v}`);
 }
}
main().catch(e=>{console.error(e);process.exit(1)});
