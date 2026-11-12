const fs=require('fs'),path=require('path'),crypto=require('crypto');
const sharp=require('C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1104-2876-0_26.1104.2876.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules/sharp');
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
async function main(){
 let receipt={};
 for(const v of ['A','B','C']){
  const svg=path.join(__dirname,`DEBR_${v}.svg`);if(!fs.existsSync(svg))continue;
  const preview=await sharp(svg).png().toBuffer();fs.writeFileSync(path.join(__dirname,`DEBR_${v}_preview.png`),preview);
  const png=path.join(__dirname,`DEBR_${v}_refined.png`);
  await sharp(svg,{density:144}).png().toFile(png);
  const small=await sharp(png).resize(1800,1000).png().toBuffer();
  const labels=Buffer.from(`<svg width="3660" height="100" xmlns="http://www.w3.org/2000/svg"><rect width="3660" height="100" fill="white"/><text x="900" y="44" font-family="Times New Roman" font-size="32" text-anchor="middle" fill="#193247">${v} · Editable SVG (embedded simulation frames)</text><text x="2760" y="44" font-family="Times New Roman" font-size="32" text-anchor="middle" fill="#193247">${v} · Refined PNG (3600 × 2000)</text><text x="1830" y="84" font-family="Times New Roman" font-size="23" text-anchor="middle" fill="#536A7B">One SVG master · Matching composition, images and typography</text></svg>`);
  await sharp({create:{width:3660,height:1100,channels:4,background:'white'}}).composite([{input:preview,left:0,top:0},{input:small,left:1860,top:0},{input:labels,left:0,top:1000}]).png().toFile(path.join(__dirname,`DEBR_${v}_comparison.png`));
  receipt[v]={svg_sha256:sha(svg),png_sha256:sha(png),master:'SVG',png_pixels:[3600,2000]};console.log(`Rendered ${v}`);
 }
 fs.writeFileSync(path.join(__dirname,'RENDER_RECEIPT.json'),JSON.stringify(receipt,null,2)+'\n');
}
main().catch(e=>{console.error(e);process.exit(1)});
