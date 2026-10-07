import numpy as np, mlx.core as mx
from q8common import *
rng=np.random.default_rng(13)
n=0
for dt in (mx.bfloat16, mx.float16):
    for (M,N,K) in ((33,128,512),(257,256,768),(100,128,1024),(130,128,256)):
        codes=rng.integers(0,256,(N,K),dtype=np.uint8); codes[:,:8]=[0,127,128,255,1,254,129,126]
        w=mx.array(codes.view('<u4').copy()); s=mx.array(rng.uniform(.001,.03,(N,K//64)).astype(np.float32)).astype(dt); b=mx.array(rng.uniform(-2,2,(N,K//64)).astype(np.float32)).astype(dt)
        x=mx.array(rng.standard_normal((M,K)).astype(np.float32)).astype(dt); sg,bg=group_major(s,b)
        for am in (0,1):
            for v in (800,801,802,803,804,805,806):
                if N % TILES[v][1]: continue
                y1=q8a8(x,w,sg,bg,am,v,1); y2=q8a8(x,w,s,b,am,v,2); mx.eval(y1,y2); n+=1
                assert np.array_equal(np.array(y1.astype(mx.float32)),np.array(y2.astype(mx.float32))),(dt,M,N,K,am,v)
print("staged == group-major, bit-exact in",n,"cases")
