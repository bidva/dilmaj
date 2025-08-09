# VS Code Debug Configuration Guide

This project has been configured with VS Code launch configurations and tasks to make development easier.

## Prerequisites

1. **Install Dependencies**: Run the "Install Dependencies" task or use the terminal:
   ```bash
   poetry install
   ```

2. **Set up Environment Variables**: 
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to the `.env` file:
     ```
     OPENAI_API_KEY=your_actual_api_key_here
     ```

## Debug Configurations

### 1. PDF Translator - Debug with Sample PDF
- **Purpose**: Debug the application with a sample PDF file
- **Setup**: Place a PDF file named `sample.pdf` in the project root
- **Features**: 
  - Processes only pages 1-3 for quick testing
  - Uses single concurrent request to avoid rate limiting during debugging
  - Verbose output enabled
  - Output goes to `debug_output/` directory

### 2. PDF Translator - Custom PDF Path
- **Purpose**: Debug with any PDF file path
- **Interactive**: VS Code will prompt you for:
  - PDF file path
  - GPT model to use (gpt-3.5-turbo, gpt-4, gpt-4-turbo-preview)
  - Start page number
  - End page number (leave empty for last page)

### 3. PDF Translator - Run Tests
- **Purpose**: Run and debug unit tests
- **Features**: Runs all tests in the `tests/` directory with verbose output

### 4. PDF Translator - Current File
- **Purpose**: Debug the currently open Python file
- **Use case**: For testing individual modules or scripts

## Setting Breakpoints

1. **In the processor module**: Set breakpoints in `/pdf_translator/processor.py` at key locations:
   - Line 40: `self.llm = ChatOpenAI(...)` - To debug LLM initialization
   - Line 75: `reader = PdfReader(str(pdf_path))` - To debug PDF reading
   - Line 142: `response = await self.llm.ainvoke([message])` - To debug GPT API calls
   - Line 186: `results = await asyncio.gather(*tasks, return_exceptions=True)` - To debug concurrent processing

2. **In the CLI module**: Set breakpoints in `/pdf_translator/cli.py`:
   - Line 105: `api_key = validate_api_key()` - To debug API key validation
   - Line 147: `pages = processor.extract_pages(pdf_path)` - To debug page extraction
   - Line 158: `results = processor.process_pages_async(...)` - To debug main processing

## Available Tasks

Access these through VS Code's Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) → "Tasks: Run Task":

### Build Tasks
- **Install Dependencies**: Install all project dependencies using Poetry
- **Format Code**: Format code using Black
- **Sort Imports**: Sort imports using isort
- **Lint Code**: Check code quality with flake8
- **Type Check**: Run type checking with mypy
- **Build and Lint All**: Run all the above tasks in sequence

### Test Tasks
- **Run Tests**: Execute all tests with pytest
- **Run Tests with Coverage**: Run tests and generate coverage report

### Run Tasks
- **Run PDF Translator - Sample**: Execute the CLI with sample.pdf (for quick testing)

## Common Debugging Scenarios

### 1. API Key Issues
- Set breakpoint at `validate_api_key()` in cli.py
- Check that the `.env` file exists and contains the correct API key

### 2. PDF Processing Issues
- Set breakpoint at `extract_pages()` in processor.py
- Check the PDF file path and permissions
- Verify page range parameters

### 3. GPT API Issues
- Set breakpoint at `_process_single_page()` in processor.py
- Check rate limiting and concurrent request settings
- Monitor API responses and error handling

### 4. Async Processing Issues
- Set breakpoint at `_process_pages_batch()` in processor.py
- Check semaphore limits and task creation
- Monitor exception handling in gather operation

## Tips

1. **Use the integrated terminal**: All configurations use the integrated terminal for better debugging experience
2. **Check environment variables**: Ensure `OPENAI_API_KEY` is set in your environment
3. **Monitor rate limits**: Use single concurrent request (`--concurrent 1`) when debugging to avoid API rate limits
4. **Check output directories**: Results are saved to specified output directories, check file permissions
5. **Use verbose mode**: Enable verbose logging (`--verbose`) for detailed debugging information

## Troubleshooting

If you encounter issues:

1. **Python interpreter not found**: VS Code should automatically use the virtual environment. If not, select the correct interpreter manually.
2. **Module not found**: Ensure `PYTHONPATH` is set correctly (handled automatically by the configurations).
3. **Permission errors**: Check file and directory permissions for input PDFs and output directories.
4. **API errors**: Verify your OpenAI API key and check rate limits.
