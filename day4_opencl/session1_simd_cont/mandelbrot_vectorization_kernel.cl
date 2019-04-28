//These are defined at compile time -- when the kernel is compiled, see the mandelbrot_vectorization.cpp code for details
//#define DIMX (64)
//#define DIMY (64)
//#define X_STEP (0.5f/DIMX)
//#define Y_STEP (0.4f/(DIMY/2))

__kernel void mandelbrot(__global int* map)
{
    int i,j;
    float x = -1.8f;
    for (i=0;i<DIMX;i++) {
        float y = -0.2f;
        for (j=0;j<DIMY/2;j++) {
            int iter = 0;
            float sx = x;
            float sy = y;
            while (iter < 256){
                if (sx*sx + sy*sy >= 4.0f){
                    break;
                }
                float old_sx = sx;
                sx = x + sx*sx - sy*sy;
                sy = y + 2*old_sx*sy;
                iter++;
            }
            map[i*DIMY+j] = iter;
            y+=Y_STEP;
        }
        x+=X_STEP;
    }
}

__kernel void mandelbrot_vectorized(__global int* map)
{
    const float4 xstep = (float4)(X_STEP,X_STEP,X_STEP,X_STEP);
    const float4 ystep = (float4)(Y_STEP,Y_STEP,Y_STEP,Y_STEP);
    const float4 four_step = (float4)(Y_STEP*4,Y_STEP*4,Y_STEP*4,Y_STEP*4);
    const float4 init_y = (float4)(0,1*Y_STEP,2*Y_STEP,3*Y_STEP);
    const float4 four_vec = (float4)(4.0,4.0,4.0,4.0);
    const float4 two_vec = (float4)(2.0,2.0,2.0,2.0);
    const int4 zero_ivec = (int4)(0,0,0,0);
    const int4 one_ivec = (int4)(1,1,1,1);

    float4 x = (float4)(-1.8f,-1.8f,-1.8f,-1.8f);
    for (int i=0;i<DIMX;i++) {
        float4 y = (float4)(-0.2f,-0.2f,-0.2f,-0.2f)+init_y;
        for (int j = 0; j < DIMY/2; j+=4) {
            int4 iter = zero_ivec;
            float4 sx = x;
            float4 sy = y;
            int scalar_iter = 0;
            while (scalar_iter < 256){
                float4 old_sx = sx;
                int4 vmask = isgreaterequal(sx*sx + sy*sy, four_vec);
                if (all(vmask)){
                    break;
                }
                if (all(vmask==(int4)(0,0,0,0))){
                    sx = x + sx*sx - sy*sy;
                    sy = y + two_vec*old_sx*sy;
                    iter += one_ivec;
                }else{
                    iter = select(iter+one_ivec,iter,vmask);
                }
                scalar_iter++;
            }

            //Note indivdual elements of the vector can be accessed, but this will be the performance blocker -- map[i*DIMY+j+0] = iter.x; ... map[i*DIMY+j+3] = iter.w;

            vstore4(iter,0,map+i*DIMY+j);

            y+=four_step;
        }
        x+=xstep;
    }
}


