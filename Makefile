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

test_training_locally:
	python dummy_model/training/typer_functions_training.py trigger_experiment \
															generate_datasets \
															log_training_dataset \
															preprocess_fit \
															log_preprocessing_artifacts \
															preprocess_transform \
															train_model \
															log_model_artifacts \
															evaluate_model

test_inference_locally:
	python dummy_model/batch_inference/typer_functions_inference.py load_input \
															preprocess_data \
															run_inference \
															postprocess_data \
															push_output