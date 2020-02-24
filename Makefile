setup: setup.py
	( \
		. env/bin/activate; \
		python setup.py sdist bdist_wheel; \
	)

README.md: README.org
	pandoc -f org -t markdown --output=README.md README.org
