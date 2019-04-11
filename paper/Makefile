#all: main.pdf
all: main.pdf

MAINFILES := main.tex $(shell find sections -iname '*tex')

IMAGEFILES := $(shell find -iname '*svg')

IMAGEPDFS := $(IMAGEFILES:%.svg=%.pdf)

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
main.pdf
