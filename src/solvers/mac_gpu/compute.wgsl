// GPU-owned particle inventories and MAC face velocities. All float reductions are
// gathers; only linked-list insertion uses atomics. No float-atomic extension.
struct Particle {
    p: vec4<f32>,
    // xyz, phase: water=0, grain=1, drained=2, unused=3
    v: vec4<f32>,
    q: vec4<f32>,
    // water volume / absorbed volume, solute / fast pool, heat, slow pool
    m: vec4<f32>,
    // dry volume, initial mass, absorption capacity, insoluble mass
    e: vec4<f32>,
    // retained solute
}
struct Link {
    next: u32, particle: u32, weight: f32, pad: u32
}
struct Cell {
    liquid: vec4<f32>,
    // water volume, solute, heat, heat capacity
    momentum: vec4<f32>,
    // liquid momentum, grain absorption demand
    solid: vec4<f32>,
    // external solid volume, grain heat, grain capacity, unused
    release: vec4<f32>,
    // released mass, heat, thermal exchange, allocation scale
    transfer: vec4<f32>,
    // released momentum
    equation: vec4<f32>,
    // rhs, diagonal
}
struct Face {
    mass: vec2<f32>, volume: vec2<f32>,
    velocity: vec2<f32>, mobility: vec2<f32>,
    // ww,ws
    extra: vec4<f32>,
    // ss, coefficient, pre-force water velocity, pre-force grain velocity
}
struct Solid {
    a: vec4<f32>, b: vec4<f32>, c: vec4<f32>, d: vec4<f32>
}
struct Params {
    dims: vec4<u32>,
    // xyz, particle capacity
    origin: vec4<f32>,
    // xyz,h
    high: vec4<f32>,
    // xyz, dt
    gravity: vec4<f32>,
    // xyz,rho
    physics: vec4<f32>,
    // grain diameter, drag scale, swelling, capillary
    rates: vec4<f32>,
    // absorb, extract, fast, slow
    thermal: vec4<f32>,
    // cp water, cp grain, kappa, enabled
    kinetics: vec4<f32>,
    // Ea/R, Tref, dref, uhalf
    flags: vec4<u32>,
    // open bottom, frozen bed, solid count, seed
    misc: vec4<f32>,
    // c_sat, grain friction, floor friction, maxdt
    inlet: vec4<f32>,
    // position, volume per frame
    jet: vec4<f32>,
    // velocity, nozzle radius
    control: vec4<f32>,
    // frame dt, nominal particle volume, pour T, CFL
    counts: vec4<u32>,
    // grain count (fixed prefix)
}
@group(0) @binding(0) var<uniform> cfg: Params;
@group(0) @binding(1) var<storage,read_write> src: array<Particle>;
@group(0) @binding(2) var<storage,read_write> dst: array<Particle>;
@group(0) @binding(3) var<storage,read_write> heads: array<atomic<u32>>;
@group(0) @binding(4) var<storage,read_write> links: array<Link>;
@group(0) @binding(5) var<storage,read_write> cells: array<Cell>;
@group(0) @binding(6) var<storage,read_write> faces: array<Face>;
@group(0) @binding(7) var<storage,read_write> pressure: array<f32>;
@group(0) @binding(8) var<storage,read> solids: array<Solid>;
@group(0) @binding(9) var<storage,read_write> positions: array<vec4<f32>>;
@group(0) @binding(10) var<storage,read_write> velocities: array<vec4<f32>>;
@group(0) @binding(11) var<storage,read_write> phases: array<u32>;
@group(0) @binding(12) var<storage,read_write> chemistry: array<vec4<f32>>;
@group(0) @binding(13) var<storage,read_write> moisture: array<f32>;
@group(0) @binding(14) var<storage,read_write> stats: array<vec4<f32>>;
const NONE: u32 = 0xffffffffu;
fn nc() -> u32 {
    return cfg.dims.x*cfg.dims.y*cfg.dims.z;
}
fn fd(a: u32) -> vec3<u32> {
    var d=cfg.dims.xyz;
    d[a]+=1u;
    return d;
}
fn nf(a: u32) -> u32 {
    let d=fd(a);
    return d.x*d.y*d.z;
}
fn offset(a: u32) -> u32 {
    if a==0u {
        return 0u;
    }
    if a==1u {
        return nf(0u);
    }
    return nf(0u)+nf(1u);
}
fn count_faces() -> u32 {
    return nf(0u)+nf(1u)+nf(2u);
}
fn coord(i: u32,d: vec3<u32>) -> vec3<u32> {
    return vec3(i%d.x,(i/d.x)%d.y,i/(d.x*d.y));
}
fn index(c: vec3<u32>,d: vec3<u32>) -> u32 {
    return (c.z*d.y+c.y)*d.x+c.x;
}
fn axis(fi: u32) -> u32 {
    if fi<nf(0u) {
        return 0u;
    }
    if fi<offset(2u) {
        return 1u;
    }
    return 2u;
}
fn neighbors(fi: u32) -> vec2<u32> {
    let a=axis(fi);
    let c=coord(fi-offset(a),fd(a));
    var l=NONE;
    var r=NONE;
    if c[a]>0u {
        var q=c;
        q[a]-=1u;
        l=index(q,cfg.dims.xyz);
    }
    if c[a]<cfg.dims[a] {
        r=index(c,cfg.dims.xyz);
    }
    return vec2(l,r);
}
fn cellface(i: u32,k: u32) -> u32 {
    let a=k/2u;
    var c=coord(i,cfg.dims.xyz);
    c[a]+=k%2u;
    return offset(a)+index(c,fd(a));
}
fn mass(p: Particle) -> f32 {
    if p.p.w==1.0 {
        return p.m.w+p.q.y+p.q.w+p.e.x+cfg.gravity.w*p.q.x;
    }
    return cfg.gravity.w*p.q.x+p.q.y;
}
fn capacity(p: Particle) -> f32 {
    if p.p.w==1.0 {
        return (mass(p)-cfg.gravity.w*p.q.x)*cfg.thermal.y+cfg.gravity.w*p.q.x*cfg.thermal.x;
    }
    return cfg.gravity.w*p.q.x*cfg.thermal.x+p.q.y*cfg.thermal.y;
}
fn wet(p: Particle) -> f32 {
    return clamp(p.q.x/max(p.m.z,1e-20),0.0,1.0);
}
fn volume(p: Particle) -> f32 {
    if p.p.w==1.0 {
        return p.m.x*(1.0+cfg.physics.z*wet(p));
    }
    return p.q.x;
}
fn demand(p: Particle) -> f32 {
    return max(0.0,p.m.z-p.q.x)*(1.0-exp(-max(cfg.rates.x,0.0)*stats[0].x));
}
fn unit(v: vec3<f32>) -> vec3<f32> {
    return v/max(length(v),1e-20);
}
// Shared analytic SDF contract: positive cavity, species masks, open top/outlet.
fn cavity(s: Solid,p: vec3<f32>) -> vec4<f32> {
    let rel=p.xz-s.a.xy;
    let r=length(rel);
    let rh=rel/max(r,1e-8);
    let floor=s.b.x;
    let top=s.b.y;
    if p.y>top {
        return vec4(0.0,1.0,0.0,1e20);
    }
    if s.a.z==0.0 {
        if s.c.w>0.0 && p.y<floor {
            return vec4(0.0,-1.0,0.0,1e20);
        }
        let a=vec2(max(s.b.z-s.c.x,s.c.y),floor);
        let b=vec2(max(s.b.w-s.c.x,s.c.y),top);
        let ab=b-a;
        let t=clamp(dot(vec2(r,p.y)-a,ab)/max(dot(ab,ab),1e-20),0.0,1.0);
        let delta=vec2(r,p.y)-(a+t*ab);
        let inner=max(mix(s.b.z,s.b.w,(p.y-floor)/max(top-floor,1e-8))-s.c.x,s.c.y);
        let inside=p.y>=floor && r<=inner;
        let sign=select(-1.0,1.0,inside);
        var n=delta*sign/max(length(delta),1e-8);
        if length(delta)<1e-8 {
            n=normalize(vec2(-ab.y,ab.x));
        }
        return vec4(unit(vec3(n.x*rh.x,n.y,n.x*rh.y)),sign*length(delta));
    }
    var side=s.b.z-r;
    var n=vec3(-rh.x,0.0,-rh.y);
    if s.a.z==2.0 {
        side=1e20;
        for(var k=0u;k<u32(s.c.z);k++) {
            let ang=6.2831853*f32(k)/s.c.z;
            let nk=vec2(cos(ang),sin(ang));
            let dist=s.b.z-dot(rel,nk);
            if dist<side {
                side=dist;
                n=vec3(-nk.x,0.0,-nk.y);
            }
        }
    }
    if p.y-floor<side {
        return vec4(0.0,1.0,0.0,p.y-floor);
    }
    return vec4(n,side);
}
fn sdf(p: vec3<f32>,phase: u32) -> vec4<f32> {
    var best=vec4(0.0,1.0,0.0,1e20);
    for(var i=0u;i<cfg.flags.z;i++) {
        if (u32(solids[i].a.w)&(1u<<phase))==0u {
            continue;
        }
        let v=cavity(solids[i],p);
        if v.w<best.w {
            best=v;
        }
    }
    return best;
}
fn collide(p0: Particle) -> Particle {
    var p=p0;
    let grain=p.p.w==1.0;
    let radius=select(0.0,pow(max(3.0*volume(p)/12.5663706,0.0),1.0/3.0),grain);
    for(var a=0u;a<3u;a++) {
        let before=p.v[a];
        let lo=cfg.origin[a]+radius;
        let hi=cfg.high[a]-radius-1e-6;
        if p.p[a]<lo && !(a==1u && cfg.flags.x!=0u && !grain) {
            p.p[a]=lo;
            p.v[a]=max(0.0,p.v[a]);
        }
        if p.p[a]>hi {
            p.p[a]=hi;
            p.v[a]=min(0.0,p.v[a]);
        }
        if grain {
            var tangent=p.v.xyz;
            tangent[a]=0.0;
            p.v=vec4(p.v.xyz-unit(tangent)*min(length(tangent),cfg.misc.z*abs(p.v[a]-before)),0.0);
        }
    }
    for(var k=0u;k<4u;k++) {
        let hit=sdf(p.p.xyz,u32(p.p.w));
        if hit.w>=radius {
            break;
        }
        p.p=vec4(p.p.xyz+(radius-hit.w+1e-6)*hit.xyz,p.p.w);
        let vn=min(dot(p.v.xyz,hit.xyz),0.0);
        p.v=vec4(p.v.xyz-vn*hit.xyz,0.0);
        if grain {
            var friction=0.0;
            var nearest=1e20;
            for(var j=0u;j<cfg.flags.z;j++) {
                if (u32(solids[j].a.w)&2u)==0u {
                    continue;
                }
                let d=cavity(solids[j],p.p.xyz).w;
                if d<nearest {
                    nearest=d;
                    friction=solids[j].d.x;
                }
            }
            let tangent=p.v.xyz-dot(p.v.xyz,hit.xyz)*hit.xyz;
            p.v=vec4(p.v.xyz-unit(tangent)*min(length(tangent),-vn*friction),0.0);
        }
    }
    if !grain && cfg.flags.x!=0u && p.p.y<cfg.origin.y {
        p.p.w=2.0;
    }
    return p;
}
// Each particle has 8 partitioned entries at centers and at each MAC face lattice.
fn stencil(p: vec3<f32>,group: u32,corner: u32) -> vec2<f32> {
    var d=cfg.dims.xyz;
    var q=(p-cfg.origin.xyz)/cfg.origin.w-vec3(0.5);
    var off=0u;
    if group>0u {
        let a=group-1u;
        d[a]+=1u;
        q[a]+=0.5;
        off=nc()+offset(a);
    }
    let base=vec3<i32>(floor(q));
    let t=fract(q);
    var sum=0.0;
    var result=vec2(-1.0,0.0);
    for(var k=0u;k<8u;k++) {
        let bit=vec3(k&1u,(k>>1u)&1u,(k>>2u)&1u);
        let c=base+vec3<i32>(bit);
        if any(c<vec3(0)) || any(c>=vec3<i32>(d)) {
            continue;
        }
        let wt=mix(vec3(1.0)-t,t,vec3<f32>(bit));
        let w=wt.x*wt.y*wt.z;
        sum+=w;
        if k==corner {
            result=vec2(f32(off+index(vec3<u32>(c),d)),w);
        }
    }
    result.y/=max(sum,1e-20);
    return result;
}
@compute @workgroup_size(64) fn clear(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;
    if i<nc()+count_faces() {
        atomicStore(&heads[i],NONE);
    }
    if i<nc() {
        pressure[i]=0.0;
    }
}
@compute @workgroup_size(64) fn bin(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;
    if i>=cfg.dims.w || src[i].p.w>1.0 {
        return;
    }
    for(var g=0u;g<4u;g++) {
        for(var k=0u;k<8u;k++) {
            let w=stencil(src[i].p.xyz,g,k);
            if w.x<0.0 || w.y<=0.0 {
                continue;
            }
            let li=i*32u+g*8u+k;
            links[li]=Link(atomicExchange(&heads[u32(w.x)],li),i,w.y,0u);
        }
    }
}
@compute @workgroup_size(64) fn inventory(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;
    if i>=nc() {
        return;
    }
    var c:Cell;
    var li=atomicLoad(&heads[i]);
    loop {
        if li==NONE {
            break;
        }
        let l=links[li];
        let p=src[l.particle];
        let w=l.weight;
        if p.p.w==0.0 {
            c.liquid+=w*vec4(p.q.xyz,capacity(p));
            c.momentum+=vec4(w*mass(p)*p.v.xyz,0.0);
        }
        else {
            c.solid+=w*vec4(volume(p),p.q.z,capacity(p),0.0);
            c.momentum.w+=w*demand(p);
        }
        li=l.next;
    }
    cells[i]=c;
}
@compute @workgroup_size(64) fn absorb(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;
    if i>=cfg.dims.w {
        return;
    }
    var p=src[i];
    if p.p.w>1.0 {
        dst[i]=p;
        return;
    }
    var gained=vec4(0.0);
    var mom=vec3(0.0);
    var fraction=0.0;
    for(var k=0u;k<8u;k++) {
        let w=stencil(p.p.xyz,0u,k);
        if w.x<0.0 || w.y<=0.0 {
            continue;
        }
        let c=cells[u32(w.x)];
        let take=min(c.liquid.x,c.momentum.w);
        if take<=0.0 {
            continue;
        }
        if p.p.w==0.0 {
            fraction+=w.y*take/max(c.liquid.x,1e-20);
        }
        else {
            let v=take*w.y*demand(p)/max(c.momentum.w,1e-20);
            gained+=vec4(v,v*c.liquid.yz/max(c.liquid.x,1e-20),0.0);
            mom+=v*c.momentum.xyz/max(c.liquid.x,1e-20);
        }
    }
    if p.p.w==0.0 {
        p.q*=max(0.0,1.0-fraction);
    }
    else {
        let old=mass(p)*p.v.xyz;
        p.q.x+=gained.x;
        p.e.x+=gained.y;
        p.q.z+=gained.z;
        p.v=vec4((old+mom)/max(mass(p),1e-20),0.0);
    }
    dst[i]=p;
}
fn request(p:Particle,c:Cell) -> vec3<f32> {
    let concentration=c.liquid.y/max(cfg.gravity.w*c.liquid.x+c.liquid.y,1e-20);
    let u=length(c.momentum.xyz/max(cfg.gravity.w*c.liquid.x+c.liquid.y,1e-20)-p.v.xyz);
    let t=p.q.z/max(capacity(p),1e-20);
    let arr=exp(clamp(-cfg.kinetics.x*(1.0/max(t,1e-6)-1.0/max(cfg.kinetics.y,1e-6)),-30.0,30.0));
    let rate=max(0.0,cfg.rates.y)*(1.0+u/(u+max(cfg.kinetics.w,1e-6)))*arr*cfg.kinetics.z/cfg.physics.x*clamp(1.0-concentration/max(cfg.misc.x,1e-6),0.0,1.0);
    let excess=max(0.0,p.e.x-concentration/max(1.0-concentration,1e-12)*cfg.gravity.w*p.q.x);
    return vec3(p.q.y,p.q.w,excess)*(vec3(1.0)-exp(-rate*vec3(cfg.rates.z,cfg.rates.w,cfg.rates.w)*stats[0].x));
}
@compute @workgroup_size(64) fn dissolution(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;
    if i>=nc() {
        return;
    }
    var c=cells[i];
    if c.liquid.x<=1e-20 {
        return;
    }
    var total=0.0;
    var heat=0.0;
    var mom=vec3(0.0);
    var li=atomicLoad(&heads[i]);
    loop {
        if li==NONE {
            break;
        }
        let l=links[li];
        let p=src[l.particle];
        if p.p.w==1.0 {
            let req=request(p,c)*l.weight;
            let amount=req.x+req.y+req.z;
            total+=amount;
            heat+=amount*cfg.thermal.y*p.q.z/max(capacity(p),1e-20);
            mom+=amount*p.v.xyz;
        }
        li=l.next;
    }
    let cs=clamp(cfg.misc.x,0.0,0.99);
    let room=max(0.0,cs/(1.0-cs)*cfg.gravity.w*c.liquid.x-c.liquid.y);
    let scale=min(1.0,room/max(total,1e-20));
    c.release=vec4(total*scale,heat*scale,0.0,scale);
    c.transfer=vec4(mom*scale,0.0);
    cells[i]=c;
}
@compute @workgroup_size(64) fn dissolve(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;
    if i>=cfg.dims.w {
        return;
    }
    var p=src[i];
    if p.p.w>1.0 {
        dst[i]=p;
        return;
    }
    var lost=vec3(0.0);
    var gain=vec2(0.0);
    var mom=vec3(0.0);
    for(var k=0u;k<8u;k++) {
        let w=stencil(p.p.xyz,0u,k);
        if w.x<0.0 || w.y<=0.0 {
            continue;
        }
        let c=cells[u32(w.x)];
        if c.liquid.x<=1e-20 {
            continue;
        }
        if p.p.w==1.0 {
            lost+=w.y*request(p,c)*c.release.w;
        }
        else {
            let share=w.y*p.q.x/c.liquid.x;
            gain+=share*c.release.xy;
            mom+=share*c.transfer.xyz;
        }
    }
    if p.p.w==1.0 {
        let amount=lost.x+lost.y+lost.z;
        p.q.z-=amount*cfg.thermal.y*p.q.z/max(capacity(p),1e-20);
        p.q.y-=lost.x;
        p.q.w-=lost.y;
        p.e.x-=lost.z;
    }
    else {
        let old=mass(p)*p.v.xyz;
        p.q.y+=gain.x;
        p.q.z+=gain.y;
        p.v=vec4((old+mom)/max(mass(p),1e-20),0.0);
    }
    dst[i]=p;
}
@compute @workgroup_size(64) fn heat(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;
    if i>=cfg.dims.w {
        return;
    }
    var p=src[i];
    if p.p.w>1.0 || cfg.thermal.w==0.0 {
        dst[i]=p;
        return;
    }
    var gain=0.0;
    for(var k=0u;k<8u;k++) {
        let w=stencil(p.p.xyz,0u,k);
        if w.x<0.0 || w.y<=0.0 {
            continue;
        }
        let c=cells[u32(w.x)];
        let cw=c.liquid.w;
        let cg=c.solid.z;
        if cw<=0.0 || cg<=0.0 || cw+cg<=1e-6 {
            continue;
        }
        let delta=c.solid.y/cg-c.liquid.z/cw;
        let cap=cw*cg/(cw+cg)*delta;
        let q=clamp(max(0.0,cfg.thermal.z)*delta*stats[0].x,min(cap,0.0),max(cap,0.0));
        gain+=select(q*w.y*capacity(p)/cw,-q*w.y*capacity(p)/cg,p.p.w==1.0);
    }
    p.q.z+=gain;
    dst[i]=p;
}
fn suction(i:u32) -> f32 {
    if i==NONE {
        return 0.0;
    }
    let c=cells[i];
    if c.solid.x<=0.0 {
        return 0.0;
    }
    return -cfg.physics.w*(1.0-clamp(c.liquid.x/max(pow(cfg.origin.w,3.0)-c.solid.x,1e-12),0.0,1.0));
}
@compute @workgroup_size(64) fn scatter_faces(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;
    if i>=count_faces() {
        return;
    }
    let a=axis(i);
    let adj=neighbors(i);
    let h=cfg.origin.w;
    let dt=stats[0].x;
    var f:Face;
    var mom=vec2(0.0);
    var li=atomicLoad(&heads[nc()+i]);
    loop {
        if li==NONE {
            break;
        }
        let l=links[li];
        let p=src[l.particle];
        let phase=u32(p.p.w);
        f.mass[phase]+=l.weight*mass(p);
        f.volume[phase]+=l.weight*volume(p);
        mom[phase]+=l.weight*mass(p)*p.v[a];
        li=l.next;
    }
    // Preserve the mass-weighted velocity before forces and constraints. Water
    // gathers the grid increment (FLIP), not a replacement grid average (PIC).
    f.extra=vec4(f.extra.xy,mom/max(f.mass,vec2(1e-20)));
    let phi=clamp(1.0-f.volume.y/(h*h*h),0.02,0.999);
    let perm=cfg.physics.x*cfg.physics.x*phi*phi*phi/(150.0*pow(max(1.0-phi,1e-4),2.0));
    var drag=dt*f.mass.x*cfg.physics.y/max(perm,1e-9);
    if f.mass.x<=0.0 || f.mass.y<=0.0 {
        drag=0.0;
    }
    var pos=vec3<f32>(coord(i-offset(a),fd(a)))+vec3(0.5);
    pos[a]-=0.5;
    pos=cfg.origin.xyz+h*pos;
    let boundary=adj.x==NONE || adj.y==NONE;
    let drain=cfg.flags.x!=0u && a==1u && adj.x==NONE;
    let w=f.mass.x>0.0 && !(boundary && !drain) && sdf(pos,0u).w>=0.0;
    let s=f.mass.y>0.0 && !boundary && cfg.flags.y==0u && sdf(pos,1u).w>=0.0;
    if w && s {
        let det=f.mass.x*f.mass.y+drag*(f.mass.x+f.mass.y);
        f.mobility=vec2(f.mass.y+drag,drag)/det;
        f.extra.x=(f.mass.x+drag)/det;
    }
    else if w {
        f.mobility.x=1.0/(f.mass.x+drag);
    }
    else if s {
        f.extra.x=1.0/(f.mass.y+drag);
    }
    f.volume*=h*h*h/max(f.volume.x+f.volume.y,1e-20);
    var force=f.mass*cfg.gravity[a];
    if f.mass.x>0.0 && f.mass.y>0.0 {
        let cap=-f.volume.x/h*(suction(adj.y)-suction(adj.x));
        force+=vec2(cap,-cap);
    }
    let rhs=mom+dt*force;
    f.velocity=vec2(dot(f.mobility,rhs),f.mobility.y*rhs.x+f.extra.x*rhs.y);
    let response=vec2(dot(f.mobility,f.volume),f.mobility.y*f.volume.x+f.extra.x*f.volume.y);
    f.extra.y=dt/(h*h)*dot(f.volume,response);
    faces[i]=f;
}
@compute @workgroup_size(64) fn assemble(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;
    if i>=nc() {
        return;
    }
    var div=0.0;
    var diagonal=0.0;
    for(var k=0u;k<6u;k++) {
        let f=faces[cellface(i,k)];
        div+=select(-1.0,1.0,k%2u==1u)*dot(f.volume,f.velocity)/cfg.origin.w;
        diagonal+=f.extra.y;
    }
    cells[i].equation=vec4((cells[i].liquid.x+cells[i].solid.x-pow(cfg.origin.w,3.0))/max(stats[0].x,1e-20)-div,diagonal,0.0,0.0);
}
fn solve(i:u32,color:u32) {
    if i>=nc() {
        return;
    }
    let c=coord(i,cfg.dims.xyz);
    if (c.x+c.y+c.z)%2u!=color {
        return;
    }
    if cells[i].liquid.x<=1e-15 || cells[i].equation.y<=1e-20 {
        return;
    }
    var b=cells[i].equation.x;
    for(var k=0u;k<6u;k++) {
        let fi=cellface(i,k);
        let adj=neighbors(fi);
        let other=select(adj.x,adj.y,adj.x==i);
        if other!=NONE {
            b+=faces[fi].extra.y*pressure[other];
        }
    }
    pressure[i]=max(0.0,b/cells[i].equation.y);
}
@compute @workgroup_size(64) fn red(@builtin(global_invocation_id) id:vec3<u32>) {
    solve(id.x,0u);
}
@compute @workgroup_size(64) fn black(@builtin(global_invocation_id) id:vec3<u32>) {
    solve(id.x,1u);
}
@compute @workgroup_size(64) fn project(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;
    if i>=count_faces() {
        return;
    }
    let adj=neighbors(i);
    var l=0.0;
    var r=0.0;
    if adj.x!=NONE {
        l=pressure[adj.x];
    }
    if adj.y!=NONE {
        r=pressure[adj.y];
    }
    var f=faces[i];
    let response=vec2(dot(f.mobility,f.volume),f.mobility.y*f.volume.x+f.extra.x*f.volume.y);
    f.velocity-=stats[0].x/cfg.origin.w*(r-l)*response;
    faces[i]=f;
}
@compute @workgroup_size(64) fn move_particles(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;
    if i>=cfg.dims.w {
        return;
    }
    var p=src[i];
    if p.p.w>1.0 {
        dst[i]=p;
        return;
    }
    var vel=select(vec3(0.0),p.v.xyz,p.p.w==0.0);
    for(var a=0u;a<3u;a++) {
        for(var k=0u;k<8u;k++) {
            let w=stencil(p.p.xyz,a+1u,k);
            if w.x>=0.0 && w.y>0.0 {
                let face=faces[u32(w.x)-nc()];
                let phase=u32(p.p.w);
                let before=select(0.0,face.extra.z,p.p.w==0.0);
                vel[a]+=w.y*(face.velocity[phase]-before);
            }
        }
    }
    p.v=vec4(vel,0.0);
    if p.p.w!=1.0 || cfg.flags.y==0u {
        p.p=vec4(p.p.xyz+stats[0].x*vel,p.p.w);
        p=collide(p);
    }
    dst[i]=p;
}
// Jacobi contacts: symmetric pair contributions, fixed relaxation (not per-particle
// neighbor normalization, which would destroy equal/opposite impulses).
@compute @workgroup_size(64) fn contacts(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;
    if i>=cfg.dims.w {
        return;
    }
    var p=src[i];
    if p.p.w!=1.0 || cfg.flags.y!=0u {
        dst[i]=p;
        return;
    }
    let radius=pow(3.0*volume(p)/12.5663706,1.0/3.0);
    var dp=vec3(0.0);
    var dv=vec3(0.0);
    for(var j=0u;j<cfg.counts.x;j++) {
        let q=src[j];
        if j==i || q.p.w!=1.0 {
            continue;
        }
        let delta=q.p.xyz-p.p.xyz;
        let dist=length(delta);
        let overlap=radius+pow(3.0*volume(q)/12.5663706,1.0/3.0)-dist;
        if overlap<=0.0 {
            continue;
        }
        var n=unit(delta);
        if dist<1e-8 {
            n=vec3(select(-1.0,1.0,j>i),0.0,0.0);
        }
        let mi=mass(p);
        let mj=mass(q);
        let invsum=1.0/mi+1.0/mj;
        dp-=n*overlap/(invsum*mi);
        let rel=q.v.xyz-p.v.xyz;
        let normal=max(0.0,-dot(rel,n))/invsum;
        let tangent=rel-dot(rel,n)*n;
        let impulse=normal*n-unit(tangent)*min(length(tangent)/invsum,cfg.misc.y*normal);
        dv-=impulse/mi;
    }
    p.p=vec4(p.p.xyz+0.25*dp,p.p.w);
    p.v=vec4(p.v.xyz+0.25*dv,0.0);
    dst[i]=collide(p);
}
// Serial scalar reduction only: all expensive physics above is parallel. This also
// keeps the emission pool reusable without CPU readback or overwriting live water.
fn random(x:u32)->f32 {
    var v=x;
    v=(v^(v>>16u))*0x7feb352du;
    v=(v^(v>>15u))*0x846ca68bu;
    return f32(v^(v>>16u))/4294967296.0;
}
@compute @workgroup_size(1) fn begin_frame() {
    stats[0].z+=cfg.control.x;
    stats[5].w+=cfg.inlet.w;
    // Pending inlet is split into water awaiting simulation time (6.y) and
    // water already due but waiting for a free particle slot (6.z).
    stats[6].y+=cfg.inlet.w;
}
@compute @workgroup_size(1) fn timestep() {
    var speed=length(cfg.jet.xyz);
    for(var i=0u;i<cfg.dims.w;i++) {
        if src[i].p.w<=1.0 {
            speed=max(speed,length(src[i].v.xyz));
        }
    }
    let distance=cfg.control.w*min(cfg.origin.w,cfg.physics.x);
    let g=length(cfg.gravity.xyz);
    let safe=2.0*distance/max(speed+sqrt(speed*speed+2.0*g*distance),1e-12);
    let dt=min(stats[0].z,min(cfg.misc.w,safe));
    let due=stats[6].y*min(1.0,dt/max(stats[0].z,1e-20));
    stats[6].y=max(0.0,stats[6].y-due);
    stats[6].z+=due;
    stats[0].x=dt;
    stats[0].y+=dt;
    stats[0].z=max(0.0,stats[0].z-dt);
    // Volume waits in a reservoir if the reusable pool is full. No discarded inlet mass.
    let wanted=stats[6].z;
    let number=u32(ceil(wanted/max(cfg.control.y,1e-12)));
    var emitted=0u;
    if number>0u && all(cfg.inlet.xyz>=cfg.origin.xyz) && all(cfg.inlet.xyz<cfg.high.xyz) && sdf(cfg.inlet.xyz,0u).w>=0.0 {
        for(var i=0u;i<cfg.dims.w;i++) {
            if emitted>=number {
                break;
            }
            let old=src[i];
            if old.p.w==3.0 || (old.p.w==0.0 && old.q.x+old.q.y<1e-20) || old.p.w==2.0 {
                if old.p.w==2.0 {
                    stats[4]+=vec4(old.q.x,old.q.y,old.q.z,0.0);
                }
                let seq=u32(stats[0].w)+cfg.flags.w;
                stats[0].w+=1.0;
                let r=cfg.jet.w*sqrt(random(seq*2u));
                let angle=6.2831853*random(seq*2u+1u);
                var pos=cfg.inlet.xyz+vec3(r*cos(angle),0.0,r*sin(angle));
                if any(pos<cfg.origin.xyz)||any(pos>=cfg.high.xyz)||sdf(pos,0u).w<0.0 {
                    pos=cfg.inlet.xyz;
                }
                let vol=min(cfg.control.y,stats[6].z);
                var p:Particle;
                p.p=vec4(pos,0.0);
                p.v=vec4(cfg.jet.xyz,0.0);
                p.q=vec4(vol,0.0,vol*cfg.gravity.w*cfg.thermal.x*cfg.control.z,0.0);
                src[i]=p;
                stats[5].w=max(0.0,stats[5].w-vol);
                stats[6].z=max(0.0,stats[6].z-vol);
                stats[5].z+=vol;
                emitted++;
            }
        }
    }
}
@compute @workgroup_size(64) fn export_particles(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;
    if i>=cfg.dims.w {
        return;
    }
    let p=src[i];
    positions[i]=vec4(p.p.xyz,select(p.q.x,p.q.x/max(cfg.control.y,1e-20),p.p.w==0.0));
    if p.p.w>1.0 || (p.p.w==0.0 && p.q.x+p.q.y<1e-20) {
        positions[i]=vec4(1e10,1e10,1e10,0.0);
    }
    velocities[i]=p.v;
    phases[i]=select(0u,1u,p.p.w==1.0);
    moisture[i]=select(0.0,wet(p),p.p.w==1.0);
    chemistry[i]=select(vec4(p.q.y/max(mass(p),1e-20),p.q.z/max(capacity(p),1e-20),0.0,0.0),vec4(p.q.y,p.q.w,p.q.z/max(capacity(p),1e-20),0.0),p.p.w==1.0);
}
@compute @workgroup_size(1) fn reduce() {
    var water=vec4(0.0);
    var solute=vec4(0.0);
    var count=0.0;
    var cup=0.0;
    var energy=0.0;
    var mom=vec3(0.0);
    for(var i=0u;i<cfg.dims.w;i++) {
        let p=src[i];
        if p.p.w==3.0 {
            continue;
        }
        energy+=p.q.z;
        mom+=mass(p)*p.v.xyz;
        if p.p.w<=1.0 {
            count+=1.0;
        }
        if p.p.w==0.0 {
            water.x+=p.q.x;
            solute.x+=p.q.y;
            for(var j=0u;j<cfg.flags.z;j++) {
                let solid=solids[j];
                if solid.a.z>0.0 && p.p.y>=solid.b.x && p.p.y<=solid.b.y && cavity(solid,p.p.xyz).w>=0.0 {
                    cup+=p.q.x;
                    break;
                }
            }
        }
        else if p.p.w==1.0 {
            water.y+=p.q.x;
            solute.y+=p.e.x;
            solute.z+=p.q.y+p.q.w;
        }
        else {
            water.z+=p.q.x;
            solute.w+=p.q.y;
        }
    }
    water.z+=stats[4].x;
    solute.w+=stats[4].y;
    water.w=count;
    stats[1]=water;
    stats[2]=solute;
    stats[3]=vec4(mom,energy+stats[4].z);
    var residual=0.0;
    for(var i=0u;i<nc();i++) {
        if cells[i].liquid.x<=1e-15 {
            continue;
        }
        var r=cells[i].equation.x-cells[i].equation.y*pressure[i];
        for(var k=0u;k<6u;k++) {
            let fi=cellface(i,k);
            let adj=neighbors(fi);
            let other=select(adj.x,adj.y,adj.x==i);
            if other!=NONE {
                r+=faces[fi].extra.y*pressure[other];
            }
        }
        residual=max(residual,select(max(r,0.0),abs(r),pressure[i]>0.0)*stats[0].x/pow(cfg.origin.w,3.0));
    }
    stats[5].x=residual;
    if stats[6].x==0.0 && cup+water.z>0.02*max(water.x+water.z,1e-20) {
        stats[6].x=stats[0].y;
    }
    var lo=vec3(1e20);
    var hi=vec3(-1e20);
    for(var i=0u;i<cfg.counts.x;i++) {
        lo=min(lo,src[i].p.xyz);
        hi=max(hi,src[i].p.xyz);
    }
    var occupied:array<bool,16>;
    var flux:array<f32,16>;
    for(var i=0u;i<cfg.dims.w;i++) {
        let p=src[i];
        if p.p.w>1.0 {
            continue;
        }
        let xz=vec2<u32>(clamp((p.p.xz-lo.xz)/max(hi.xz-lo.xz,vec2(1e-12))*4.0,vec2(0.0),vec2(3.0)));
        let bin=xz.x+4u*xz.y;
        if p.p.w==1.0 {
            occupied[bin]=true;
        }
        else if p.p.y>=lo.y-cfg.origin.w && p.p.y<=hi.y+cfg.origin.w {
            flux[bin]+=p.q.x*max(0.0,-p.v.y);
        }
    }
    var sum=0.0;
    var n=0.0;
    for(var i=0u;i<16u;i++) {
        if occupied[i] {
            sum+=flux[i];
            n+=1.0;
        }
    }
    let mean=sum/max(n,1.0);
    var variance=0.0;
    for(var i=0u;i<16u;i++) {
        if occupied[i] {
            variance+=pow(flux[i]-mean,2.0);
        }
    }
    stats[5].y=1.0;
    if mean>1e-12 {
        stats[5].y=1.0/(1.0+sqrt(variance/max(n,1.0))/mean);
    }
}
