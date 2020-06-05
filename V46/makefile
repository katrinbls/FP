
all: main.pdf

main.pdf: main.tex header.tex literatur.bib magnetfeld.pdf Winkel_frei1.pdf Winkel_frei2.pdf Winkel_n-dotiert_1.pdf Winkel_n-dotiert_2.pdf Winkel_undotiert.pdf
	lualatex main.tex
	biber main.bcf
	lualatex main.tex

magnetfeld.pdf: magnetfeld.py magnetfeldData.txt 
	python magnetfeld.py

Winkel_frei1.pdf Winkel_frei2.pdf Winkel_n-dotiert_1.pdf Winkel_n-dotiert_2.pdf Winkel_undotiert.pdf: effektiveMasse.py
	python effektiveMasse.py

clean:
	rm Aufgabe2