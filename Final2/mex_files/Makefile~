% build mex
mex -v  -I/usr/include lsd.c
mex -v  CFLAGS='-D_GNU_SOURCE -fexceptions -fno-omit-frame-pointer -pthread -ansi -fPIC -fopenmp' LDFLAGS='-pthread -shared -fopenmp' -I/usr/include -Ilib lib/alignments.c lib/ntuple.c  lib/misc.c lib/ntuples_aux.c  lib/nfa.c  alignments_slow.c -output alignments_slow

mex -v  CFLAGS='-D_GNU_SOURCE -fexceptions -fno-omit-frame-pointer -pthread  -fPIC -fopenmp' LDFLAGS='-pthread -shared -fopenmp' -I/usr/include -Ilib lib/alignments.c lib/ntuple.c   lib/misc.c lib/ntuples_aux.c  lib/nfa.c  alignments_fast.c -output alignments_fast


