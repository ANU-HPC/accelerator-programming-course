#!/usr/bin/env python3

import pandas as pd

x = pd.read_csv('lsb.mandelbrot.r0', comment='#', sep="\s+")

y = x[(x.region == 'mandelbrot') | (x.region == 'mandelbrot_128')]

from ggplot import *
pp = ggplot(y, aes(x='time', colour = 'region')) + geom_histogram(binwidth=.1) + xlab("Time (us)") + ylab("Frequency (#)") + ggtitle("Mandelbrot times") + scale_x_discrete(expand = (0,0), limits = (0,max(y.time)+max(y.time)/4))
pp.save("runtimes.png", width=None, height=None, dpi=500)
