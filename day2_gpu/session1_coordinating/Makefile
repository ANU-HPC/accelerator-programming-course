all : keywordfinder keywordfinderV1 verne128.txt
	
% : %.c makefile
	gcc -o $@ $<

% : %.cu makefile
	nvcc -o $@ $< 

verne128.txt : verne.txt
	cat verne.txt verne.txt verne.txt verne.txt > 4.txt
	cat 4.txt 4.txt 4.txt 4.txt > 16.txt
	cat 16.txt 16.txt 16.txt 16.txt > 64.txt
	cat 64.txt 64.txt > verne128.txt
	rm 4.txt 16.txt 64.txt


