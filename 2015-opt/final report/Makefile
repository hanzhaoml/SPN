SRC=$(wildcard *.tex)
PDF=$(SRC:.tex=.pdf)

all: $(PDF)
%.pdf: %.tex *.sty *.bib
	latexmk --pdf $<
	latexmk -c
	write-good $< || true
clean: ; latexmk -C; rm -f *.bbl
