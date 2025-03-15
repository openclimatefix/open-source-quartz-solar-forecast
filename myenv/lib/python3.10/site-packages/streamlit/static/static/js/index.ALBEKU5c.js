import{n as i,k as m,r as c,j as e}from"./index.tKq1MI69.js";import{P as l,R as p}from"./RenderInPortalIfExists.DDrCBrUu.js";const d=""+new URL("../media/flake-0.DgWaVvm5.png",import.meta.url).href,f=""+new URL("../media/flake-1.B2r5AHMK.png",import.meta.url).href,S=""+new URL("../media/flake-2.BnWSExPC.png",import.meta.url).href,o=150,r=150,g=10,E=90,u=4e3,n=(t,a=0)=>Math.random()*(t-a)+a,w=()=>m(`from{transform:translateY(0)
      rotateX(`,n(360),`deg)
      rotateY(`,n(360),`deg)
      rotateZ(`,n(360),"deg);}to{transform:translateY(calc(100vh + ",o,`px))
      rotateX(0)
      rotateY(0)
      rotateZ(0);}`),x=i("img",{target:"es7rdur0"})(({theme:t})=>({position:"fixed",top:`${-o}px`,marginLeft:`${-r/2}px`,zIndex:t.zIndices.balloons,left:`${n(E,g)}vw`,animationDelay:`${n(u)}ms`,height:`${o}px`,width:`${r}px`,pointerEvents:"none",animationDuration:"3000ms",animationName:w(),animationTimingFunction:"ease-in",animationDirection:"normal",animationIterationCount:1,opacity:1})),_=100,s=[d,f,S],I=s.length,M=({particleType:t})=>e(x,{src:s[t]}),h=function({scriptRunId:a}){return e(p,{children:e(l,{className:"stSnow","data-testid":"stSnow",scriptRunId:a,numParticleTypes:I,numParticles:_,ParticleComponent:M})})},P=c.memo(h);export{_ as NUM_FLAKES,P as default};
