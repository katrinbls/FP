
all: main.pdf

main.pdf: main.tex header.tex literatur.bib 
	lualatex main.tex
	biber main.bcf
	lualatex main.tex

clean:
	rm Aufgabe2