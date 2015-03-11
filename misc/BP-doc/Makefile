BASE = BP
DEPS = $(shell perl getdep.pl < BP.tex)

all: pdf

pdf: ${BASE}.pdf

${BASE}.pdf: ${BASE}.tex ${DEPS}
	xelatex -no-pdf --interaction=nonstopmode ${BASE}
	bibtex ${BASE}
	xelatex -no-pdf --interaction=nonstopmode ${BASE}
	xelatex --interaction=nonstopmode ${BASE}

clean:
	rm -f *.aux
	rm -f *.log
	rm -f *.lot
	rm -f *.out
	rm -f *.toc
	rm -f *.blg
	rm -f *.bbl
	rm -f *.lof
	rm -f *.xdv
	rm -f *.hst
	rm -f *.ver
	rm -f *.synctex.gz
	rm -f ${BASE}.pdf

