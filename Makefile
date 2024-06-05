clean:
	find . | grep -E "(__pycache__|.pytest_cache|.ipynb_checkpoints)" | xargs rm -rf

setup_dev_env:
	@if [ -d "$$(poetry env info --path)" ]; then \
		echo "Deleting existing poetry environment..."; \
		rm -rf "$$(poetry env info --path)"; \
		echo "Existing poetry environment deleted."; \
	fi
	rm -f poetry.lock
	poetry install --all-extras
	poetry run python -m ipykernel install --user --name=dummy-model