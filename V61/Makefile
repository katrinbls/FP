all: main.pdf 

main.pdf: main.tex header.tex literatur.bib g1g2.pdf
	lualatex main.tex
	biber main.bcf
	lualatex main.tex
	lualatex main.tex

g1g2.pdf: 
	python theorie.py

clean: