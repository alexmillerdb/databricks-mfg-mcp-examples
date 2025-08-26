# Search and Convert Documentation to Markdown

You are tasked with fetching content from one or more documentation URLs and creating well-structured markdown files optimized for Claude Code context.

## Instructions

1. **Parse the Input**: The user will provide either:
   - A single URL
   - Multiple URLs (comma-separated or on separate lines)
   - A URL with an optional output filename

2. **For Each URL**:
   - Use WebFetch to retrieve the documentation content
   - Extract the main content, code examples, and important sections
   - Preserve the structure (headings, code blocks, tables, lists)
   - Identify the documentation type (API reference, tutorial, guide, etc.)

3. **Create Markdown File(s)**:
   - Generate a clean filename from the page title if not specified
   - Place files in `docs/` by default
   - Format content for Claude Code usage

4. **Output Format**:
   ```markdown
   # [Title]

   **Source**: [URL]
   **Domain**: [domain]
   **Fetched**: [date]
   **Type**: [Documentation type]

   ## Overview
   [Brief summary of the documentation]

   ## Key Concepts
   [Main content with preserved structure]

   ## Code Examples
   [Extracted code examples with proper formatting]

   ## Implementation Notes
   [Practical notes for using this in development]

   ## Related Resources
   - Reference CLAUDE.md for project guidelines
   - Check /docs folder for additional examples
   ```

5. **Special Handling**:
   - For Databricks docs: Focus on Mosaic AI, Unity Catalog, MLflow 3 content
   - For API docs: Extract endpoint details, parameters, and examples
   - For tutorials: Preserve step-by-step structure
   - For code examples: Ensure proper language detection and formatting

6. **File Naming**:
   - Remove special characters
   - Use kebab-case
   - Limit to 50 characters
   - Add topic prefix (e.g., `mlflow-`, `unity-catalog-`, `mosaic-ai-`)

## Examples

**Single URL**:
```
/search-docs https://docs.databricks.com/en/mlflow/tracking.html
```

**Multiple URLs**:
```
/search-docs https://docs.databricks.com/en/generative-ai/agent-framework/index.html, https://docs.databricks.com/en/machine-learning/llms/langchain.html
```

**With custom filename**:
```
/search-docs https://docs.databricks.com/en/data-governance/unity-catalog/index.html unity-catalog-complete-guide.md
```

## Process

1. Acknowledge the request and list the URLs to process
2. Use WebFetch with appropriate prompts to extract relevant content
3. Create markdown file(s) with the extracted content
4. Report success with file paths and brief summary of content captured

Remember to:
- Focus on technical content, code examples, and implementation details
- Remove navigation, ads, and irrelevant UI elements
- Preserve important warnings, notes, and best practices
- Add context about when this documentation was fetched
- Include notes about compatibility and version requirements when mentioned