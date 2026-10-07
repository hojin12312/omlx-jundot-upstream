import numpy as np, mlx.core as mx
from q8common import *
rng=np.random.default_rng(11)
for dt in (mx.bfloat16, mx.float16):
    for (M,N,K) in ((33,128,320),(257,256,576),(100,128,1024)):
        codes=rng.integers(0,256,(N,K),dtype=np.uint8); codes[:,:8]=[0,127,128,255,1,254,129,126]
        w=mx.array(codes.view('<u4').copy()); s=mx.array(rng.uniform(.001,.03,(N,K//64)).astype(np.float32)).astype(dt); b=mx.array(rng.uniform(-2,2,(N,K//64)).astype(np.float32)).astype(dt)
        x=mx.array(rng.standard_normal((M,K)).astype(np.float32)).astype(dt)
        sg,bg=group_major(s,b)
        for v in (800,803,806):
            y1=q8a8(x,w,sg,bg,0,v,1); y2=q8a8(x,w,s,b,0,v,2); mx.eval(y1,y2)
            assert np.array_equal(np.array(y1.astype(mx.float32)),np.array(y2.astype(mx.float32))), (dt,M,N,K,v)
print("direct == group-major, bit-exact")
