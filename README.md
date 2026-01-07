# any-llm-ts

A unified TypeScript interface for LLM providers. Supports both remote providers (OpenAI, Anthropic) and local providers (Ollama, llamafile).

Inspired by [mozilla-ai/any-llm](https://github.com/mozilla-ai/any-llm).

## Features

- 🔄 **Unified API** - Same interface for all providers
- 🌐 **Remote Providers** - OpenAI, Anthropic, Mistral, Groq
- 🏠 **Local Providers** - Ollama, llamafile
- 📡 **Streaming** - Full streaming support
- 🔧 **Tool Calling** - Function/tool calling support
- 🎯 **TypeScript** - Full type safety
- 🔌 **Extensible** - Easy to add new providers
- 📦 **Official SDKs** - Built on official provider SDKs for reliability

## Installation

```bash
npm install any-llm-ts
```

Or install from GitHub:

```bash
npm install github:YOUR_USERNAME/any-llm-ts
```

## Quick Start

### Simple API

```typescript
import { completion } from 'any-llm-ts';

// Use provider:model format
const response = await completion({
  model: 'openai:gpt-4o',
  messages: [{ role: 'user', content: 'Hello!' }],
  api_key: process.env.OPENAI_API_KEY,
});

console.log(response.choices[0].message.content);
```

### Streaming

```typescript
import { completionStream } from 'any-llm-ts';

for await (const chunk of completionStream({
  model: 'anthropic:claude-3-5-sonnet-20241022',
  messages: [{ role: 'user', content: 'Tell me a story' }],
  api_key: process.env.ANTHROPIC_API_KEY,
})) {
  process.stdout.write(chunk.choices[0].delta.content || '');
}
```

### Class-based API

For better performance when making multiple requests:

```typescript
import { AnyLLM } from 'any-llm-ts';

// Create a reusable instance
const llm = AnyLLM.create('openai', { apiKey: process.env.OPENAI_API_KEY });

// Check availability
const available = await llm.isAvailable();

// List models
const models = await llm.listModels();

// Make completions (model name without provider prefix)
const response = await llm.completion({
  model: 'gpt-4o-mini',
  messages: [{ role: 'user', content: 'Hello!' }],
});
```

### Local LLMs

```typescript
import { completion, AnyLLM } from 'any-llm-ts';

// Ollama (runs on localhost:11434)
const ollamaResponse = await completion({
  model: 'ollama:llama3.2',
  messages: [{ role: 'user', content: 'Hello!' }],
});

// llamafile (runs on localhost:8080)
const llamafileResponse = await completion({
  model: 'llamafile:default',
  messages: [{ role: 'user', content: 'Hello!' }],
});

// Check if Ollama is available
const ollama = AnyLLM.create('ollama');
if (await ollama.isAvailable()) {
  const models = await ollama.listModels();
  console.log('Available models:', models.map(m => m.id));
}
```

## Providers

### Remote Providers (require API key)

| Provider | Model Format | API Key Env Var |
|----------|--------------|-----------------|
| OpenAI | `openai:gpt-4o` | `OPENAI_API_KEY` |
| Anthropic | `anthropic:claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` |
| Mistral | `mistral:mistral-large-latest` | `MISTRAL_API_KEY` |
| Groq | `groq:llama-3.1-70b-versatile` | `GROQ_API_KEY` |

### Local Providers (no API key needed)

| Provider | Model Format | Default URL |
|----------|--------------|-------------|
| Ollama | `ollama:llama3.2` | `http://localhost:11434` |
| llamafile | `llamafile:default` | `http://localhost:8080` |

## Tool Calling

```typescript
import { completion } from 'any-llm-ts';

const response = await completion({
  model: 'openai:gpt-4o',
  messages: [{ role: 'user', content: 'What is the weather in Paris?' }],
  tools: [{
    type: 'function',
    function: {
      name: 'get_weather',
      description: 'Get the current weather in a location',
      parameters: {
        type: 'object',
        properties: {
          location: { type: 'string', description: 'City name' },
        },
        required: ['location'],
      },
    },
  }],
  api_key: process.env.OPENAI_API_KEY,
});

if (response.choices[0].message.tool_calls) {
  console.log('Tool calls:', response.choices[0].message.tool_calls);
}
```

## Custom Providers

You can add custom providers:

```typescript
import { registerProvider, BaseProvider } from 'any-llm-ts';

class MyCustomProvider extends BaseProvider {
  readonly PROVIDER_NAME = 'custom';
  readonly ENV_API_KEY_NAME = 'CUSTOM_API_KEY';
  readonly PROVIDER_DOCUMENTATION_URL = 'https://docs.custom.ai';
  readonly API_BASE = 'https://api.custom.ai/v1';
  
  constructor(config) {
    super(config);
    this.init();
  }
  
  // Implement required methods...
  async completion(request) { /* ... */ }
  async *completionStream(request) { /* ... */ }
  async isAvailable() { /* ... */ }
}

registerProvider('custom', MyCustomProvider);

// Now use it
const response = await completion({
  model: 'custom:my-model',
  messages: [{ role: 'user', content: 'Hello!' }],
});
```

## API Reference

### Functions

#### `completion(request: CompletionRequest): Promise<ChatCompletion>`

Create a chat completion (non-streaming).

#### `completionStream(request: CompletionRequest): AsyncIterable<ChatCompletionChunk>`

Create a streaming chat completion.

#### `listModels(provider: string, config?: ProviderConfig): Promise<ModelInfo[]>`

List available models for a provider.

#### `checkProvider(provider: string, config?: ProviderConfig): Promise<ProviderStatus>`

Check if a provider is available.

#### `getSupportedProviders(): string[]`

Get list of all registered provider names.

### Types

```typescript
interface CompletionRequest {
  model: string;              // Model name or provider:model format
  messages: Message[];        // Conversation messages
  provider?: string;          // Provider name (if not in model)
  tools?: Tool[];             // Function/tool definitions
  max_tokens?: number;        // Maximum tokens to generate
  temperature?: number;       // Sampling temperature (0-2)
  top_p?: number;             // Nucleus sampling parameter
  stream?: boolean;           // Enable streaming
  api_key?: string;           // API key (for remote providers)
  api_base?: string;          // Custom API base URL
}

interface Message {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | MessageContent[];
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

interface ChatCompletion {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  choices: CompletionChoice[];
  usage?: CompletionUsage;
}
```

## Error Handling

```typescript
import { 
  completion, 
  MissingApiKeyError, 
  RateLimitError,
  isAnyLLMError 
} from 'any-llm-ts';

try {
  const response = await completion({
    model: 'openai:gpt-4o',
    messages: [{ role: 'user', content: 'Hello!' }],
  });
} catch (error) {
  if (error instanceof MissingApiKeyError) {
    console.error('Please set OPENAI_API_KEY');
  } else if (error instanceof RateLimitError) {
    console.error('Rate limited, retry later');
  } else if (isAnyLLMError(error)) {
    console.error('LLM error:', error.code, error.message);
  }
}
```

## Architecture

Under the hood, any-llm-ts uses the official SDKs from each provider rather than making raw HTTP calls. This provides:

- **Better error handling** - SDK-specific errors are mapped to consistent any-llm-ts error types
- **Automatic retries** - Built-in retry logic from official SDKs
- **Type safety** - Full TypeScript support from provider SDKs
- **API versioning** - SDKs handle API version compatibility
- **Streaming** - Native streaming support from each SDK

### SDK Dependencies

| Provider | SDK Package |
|----------|-------------|
| OpenAI, Groq, Together, OpenRouter, DeepSeek, LM Studio | [`openai`](https://www.npmjs.com/package/openai) |
| Anthropic | [`@anthropic-ai/sdk`](https://www.npmjs.com/package/@anthropic-ai/sdk) |
| Ollama | [`ollama`](https://www.npmjs.com/package/ollama) |
| Llamafile | [`openai`](https://www.npmjs.com/package/openai) (OpenAI-compatible API) |

OpenAI-compatible providers (Groq, Together, OpenRouter, etc.) use the OpenAI SDK with a custom `baseURL` configuration.

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Run tests
npm test
```

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

