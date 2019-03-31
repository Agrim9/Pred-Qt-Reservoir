all: Thesis.pdf

main.pdf: Thesis.tex
	pdflatex -shell-escape Thesis.tex

.PHONY: clean
clean:
	$(RM) main.pdf \
Thesis.aux \
Thesis.bbl \
Thesis.blg \
Thesis.log \
Thesis.out \
Thesis.pdf \
Thesis.tex~
