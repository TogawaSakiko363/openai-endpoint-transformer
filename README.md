# OpenAI Responses -> Chat Completions 完整转换层

该程序是一个本地 OpenAI API 兼容转换层，主要用于以下场景：

- 上游服务平台仅提供 [`/v1/chat/completions`](main.go:269)
- 客户端平台只支持 [`/v1/responses`](main.go:230)
- 同时还希望其余 OpenAI 风格端点例如 [`/v1/models`](main.go:323) 能直接透传
- 我的实际用途：将使用New API的第三方中转站聚合到自用的Sub2API站点，因为第三方站点通常使用 `/v1/chat/completions` 风格提供各种国产模型，它也是Cluade、Gemini等模型的通用格式，但是Sub2API对于OpenAI格式的上游仅支持 `/v1/responses` 风格，本项目的最初目标就是为了解决此问题

它会将客户端发往 [`/v1/responses`](main.go:230) 的请求转换后转发到上游 [`/v1/chat/completions`](main.go:269)，再把返回结果重建成 responses API 兼容格式；其余未拦截的 [`/v1/*`](main.go:323) 端点则原样代理到上游服务。

## 版本要求

- Go 1.26

## 编译

```bash
go build -o app .
```

## 启动方式

```bash
./app -s "http://100.100.100.100:3000" -l localhost:4321
```

参数说明：

- `-s`：上游服务根地址
- `-l`：本地 HTTP 监听地址

## 转换能力

### 1. 完整代理 OpenAI 风格端点

除了 [`/v1/responses`](main.go:230) 与 [`/healthz`](main.go:229) 由本地处理外，其余 [`/v1/*`](main.go:323) 请求都会直接透传到上游，包括但不限于：

- [`/v1/models`](main.go:323)
- [`/v1/embeddings`](main.go:323)
- [`/v1/images`](main.go:323)
- [`/v1/audio`](main.go:323)
- 以及其他兼容 OpenAI 风格的自定义路径

### 2. Responses -> Chat Completions 请求转换

已支持以下请求字段映射：

- `model`
- `instructions`
- `temperature`
- `top_p`
- `max_output_tokens` -> `max_tokens`
- `metadata`
- `user`
- `presence_penalty`
- `frequency_penalty`
- `tool_choice`
- `tools`
- `parallel_tool_calls`
- `stream`

### 3. 多模态输入块映射

[`normalizeInput()`](main.go:407) 与 [`mapInputPart()`](main.go:509) 已支持常见 responses 输入块转换：

- `input_text` -> chat `text`
- `input_image` -> chat `image_url`
- `image_url` -> chat `image_url`
- `input_file` -> 文本占位映射
- `function_call` -> assistant `tool_calls`
- `function_call_output` -> tool message

### 4. 函数调用输出块重建

[`convertChatToResponses()`](main.go:588) 会把上游 chat completions 的：

- `choices[].message.content`
- `choices[].message.tool_calls`

重建为 responses 风格的：

- `output[].content[].type=output_text`
- `output[].content[].type=function_call`

### 5. SSE 流式传输转换

[`streamChatToResponses()`](main.go:672) 会把上游 chat completions SSE 增量流重组为 responses 风格事件流，当前覆盖：

- `response.created`
- `response.output_item.added`
- `response.output_text.delta`
- `response.function_call_arguments.delta`
- `response.output_item.done`
- `response.completed`
- `data: [DONE]`

## 健康检查

- [`/healthz`](main.go:229)

## 说明与边界

该实现已经从“基础示例代理”扩展为“完整转换层”方向的版本，重点解决了：

- 非 [`/v1/responses`](main.go:230) 端点完整透传
- SSE 流式传输适配
- 文本/图片/函数调用相关块的双向重建

仍需注意：

- 如果上游 chat completions 服务对多模态块格式有私有扩展，仍需按其实际字段补充 [`mapInputPart()`](main.go:509)
- `input_file` 目前只能做占位转换；若上游支持文件理解专有格式，需要按目标协议继续扩展
- 不同兼容服务厂商的 SSE chunk 字段命名可能有差异，必要时应基于抓包结果微调 [`processSSEEvent()`](main.go:719)
