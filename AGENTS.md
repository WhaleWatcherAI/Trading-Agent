# AGENTS Guidelines for Codebase

## Build, Lint, Test Commands
- Use `npm run build` to compile the project.
- Use `npm run lint` to check code style.
- Use `npm test` to run all tests.
- Use `npm test -- --watch` for continuous testing.
- To run a specific test, use `npm test -- -t 'test name'`.

## Code Style Guidelines
- Use ES6+ import syntax.
- Indent with 2 spaces consistently.
- Use camelCase for variables and functions.
- Use PascalCase for types and classes.
- Handle errors with try-catch or promise `.catch()`.
- Write clear, descriptive comments.
- Keep functions small and focused.
- Use TypeScript types for all variables and functions.
- Validate inputs and outputs.

## Cursor Rules
- Follow rules in `.cursor/rules/` and `.cursorrules` files.

## Copilot Rules
- Follow instructions in `.github/copilot-instructions.md`.

Ensure all code adheres to these standards for consistency and quality.