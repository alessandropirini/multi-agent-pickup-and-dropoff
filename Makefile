.PHONY: install run-ex14

install:
	poetry install

run-ex14:
	poetry run pdm4ar-exercise --exercise 14
