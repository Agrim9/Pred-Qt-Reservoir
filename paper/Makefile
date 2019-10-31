#all: main.pdf
all: main.pdf main-onecol.pdf

MAINFILES := main.tex

IMAGEFILES := $(shell find -iname '*svg')

IMAGEPDFS := $(IMAGEFILES:%.svg=%.pdf)

main-onecol.pdf: main.bib $(MAINFILES) $(IMAGEPDFS)
	sed '1s/main,10pt,twocolumn/main,12pt,onecolumn,draftcls/' main.tex > main-onecol.tex
	pdflatex main-onecol && bibtex main-onecol && \
pdflatex main-onecol && pdflatex main-onecol
	$(RM) main-onecol.tex

main.pdf: main.bib $(MAINFILES) $(IMAGEPDFS)
	pdflatex main.tex && bibtex main && pdflatex main && pdflatex main

%.pdf: %.svg
	inkscape --export-pdf=$@ $<


.PHONY: clean
clean:
	$(RM) $(IMAGEPDFS) main.aux \
main.bbl \
main.blg \
main.log \
main.out \
main.pdf \
main-onecol.*
