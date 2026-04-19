package main

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path"
	"strings"
	"time"
)

type ResponsesRequest struct {
	Model              string         `json:"model"`
	Input              any            `json:"input"`
	Instructions       string         `json:"instructions,omitempty"`
	Temperature        *float64       `json:"temperature,omitempty"`
	TopP               *float64       `json:"top_p,omitempty"`
	MaxOutputTokens    *int           `json:"max_output_tokens,omitempty"`
	Metadata           map[string]any `json:"metadata,omitempty"`
	User               string         `json:"user,omitempty"`
	PresencePenalty    *float64       `json:"presence_penalty,omitempty"`
	FrequencyPenalty   *float64       `json:"frequency_penalty,omitempty"`
	ToolChoice         any            `json:"tool_choice,omitempty"`
	Tools              []ResponseTool `json:"tools,omitempty"`
	ParallelToolCalls  *bool          `json:"parallel_tool_calls,omitempty"`
	Store              *bool          `json:"store,omitempty"`
	Stream             bool           `json:"stream,omitempty"`
	PreviousResponseID string         `json:"previous_response_id,omitempty"`
}

type ResponseTool struct {
	Type     string            `json:"type"`
	Name     string            `json:"name,omitempty"`
	Function *ResponseFunction `json:"function,omitempty"`
}

type ResponseFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

type ChatCompletionsRequest struct {
	Model            string         `json:"model"`
	Messages         []ChatMessage  `json:"messages"`
	Temperature      *float64       `json:"temperature,omitempty"`
	TopP             *float64       `json:"top_p,omitempty"`
	MaxTokens        *int           `json:"max_tokens,omitempty"`
	Metadata         map[string]any `json:"metadata,omitempty"`
	User             string         `json:"user,omitempty"`
	PresencePenalty  *float64       `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64       `json:"frequency_penalty,omitempty"`
	ToolChoice       any            `json:"tool_choice,omitempty"`
	Tools            []ChatTool     `json:"tools,omitempty"`
	ParallelTools    *bool          `json:"parallel_tool_calls,omitempty"`
	Stream           bool           `json:"stream,omitempty"`
}

type ChatMessage struct {
	Role       string     `json:"role"`
	Content    any        `json:"content,omitempty"`
	Name       string     `json:"name,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Refusal    any        `json:"refusal,omitempty"`
}

type ChatTool struct {
	Type     string       `json:"type"`
	Function ChatFunction `json:"function"`
}

type ChatFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type ChatCompletionResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []ChatChoice `json:"choices"`
	Usage   *Usage       `json:"usage,omitempty"`
}

type ChatChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason,omitempty"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens,omitempty"`
	CompletionTokens int `json:"completion_tokens,omitempty"`
	TotalTokens      int `json:"total_tokens,omitempty"`
}

type ResponsesAPIResponse struct {
	ID                 string           `json:"id"`
	Object             string           `json:"object"`
	CreatedAt          int64            `json:"created_at"`
	Status             string           `json:"status"`
	Model              string           `json:"model"`
	Output             []ResponseOutput `json:"output"`
	Usage              *ResponsesUsage  `json:"usage,omitempty"`
	ParallelToolCalls  *bool            `json:"parallel_tool_calls,omitempty"`
	Store              bool             `json:"store"`
	Temperature        *float64         `json:"temperature,omitempty"`
	TopP               *float64         `json:"top_p,omitempty"`
	MaxOutputTokens    *int             `json:"max_output_tokens,omitempty"`
	PreviousResponseID *string          `json:"previous_response_id,omitempty"`
	Text               ResponseText     `json:"text"`
	ToolChoice         any              `json:"tool_choice,omitempty"`
	Tools              []ResponseTool   `json:"tools,omitempty"`
	Metadata           map[string]any   `json:"metadata,omitempty"`
	User               string           `json:"user,omitempty"`
}

type ResponsesUsage struct {
	InputTokens  int `json:"input_tokens,omitempty"`
	OutputTokens int `json:"output_tokens,omitempty"`
	TotalTokens  int `json:"total_tokens,omitempty"`
}

type ResponseOutput struct {
	ID      string                `json:"id"`
	Type    string                `json:"type"`
	Status  string                `json:"status,omitempty"`
	Role    string                `json:"role,omitempty"`
	Content []ResponseOutputBlock `json:"content,omitempty"`
}

type ResponseOutputBlock struct {
	Type        string         `json:"type"`
	Text        string         `json:"text,omitempty"`
	Annotations []any          `json:"annotations,omitempty"`
	Arguments   string         `json:"arguments,omitempty"`
	CallID      string         `json:"call_id,omitempty"`
	Name        string         `json:"name,omitempty"`
	Output      any            `json:"output,omitempty"`
	Detail      string         `json:"detail,omitempty"`
	ImageURL    *ResponseImage `json:"image_url,omitempty"`
}

type ResponseImage struct {
	URL string `json:"url,omitempty"`
}

type ResponseText struct {
	Format ResponseTextFormat `json:"format"`
}

type ResponseTextFormat struct {
	Type string `json:"type"`
}

type ErrorEnvelope struct {
	Error APIError `json:"error"`
}

type APIError struct {
	Message string `json:"message"`
	Type    string `json:"type,omitempty"`
}

type StreamState struct {
	ResponseID      string
	OutputID        string
	MessageStarted  bool
	MessageFinished bool
	ToolCalls       map[int]*ToolCallState
}

type ToolCallState struct {
	ID        string
	Name      string
	Arguments strings.Builder
	Started   bool
	Finished  bool
}

type ProxyServer struct {
	client   *http.Client
	upstream *url.URL
}

func main() {
	serviceEndpoint := flag.String("s", "", "上游服务根地址，例如 http://127.0.0.1:3000")
	listenAddr := flag.String("l", "localhost:4321", "本地 HTTP 监听地址，例如 localhost:4321")
	flag.Parse()

	if *serviceEndpoint == "" {
		log.Fatal("缺少 -s 参数")
	}

	upstreamBase, err := normalizeBaseURL(*serviceEndpoint)
	if err != nil {
		log.Fatalf("无效的 -s 参数: %v", err)
	}

	proxy := &ProxyServer{
		client:   &http.Client{Timeout: 0},
		upstream: upstreamBase,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", proxy.handleHealthz)
	mux.HandleFunc("/v1/responses", proxy.handleResponses)
	mux.HandleFunc("/", proxy.handleProxy)

	server := &http.Server{
		Addr:              *listenAddr,
		Handler:           loggingMiddleware(mux),
		ReadHeaderTimeout: 15 * time.Second,
	}

	log.Printf("OpenAI endpoint transformer listening on http://%s, upstream=%s", *listenAddr, upstreamBase.String())
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}

func (p *ProxyServer) handleHealthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
}

func (p *ProxyServer) handleResponses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "only POST is supported")
		return
	}

	var req ResponsesRequest
	decoder := json.NewDecoder(r.Body)
	decoder.UseNumber()
	if err := decoder.Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", fmt.Sprintf("invalid JSON body: %v", err))
		return
	}

	chatReq, err := convertResponsesRequest(req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	payload, err := json.Marshal(chatReq)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "internal_error", err.Error())
		return
	}

	upstreamURL := p.upstream.ResolveReference(&url.URL{Path: "/v1/chat/completions"})
	upstreamReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, upstreamURL.String(), bytes.NewReader(payload))
	if err != nil {
		writeError(w, http.StatusInternalServerError, "internal_error", err.Error())
		return
	}
	copyHeaders(upstreamReq.Header, r.Header)
	upstreamReq.Header.Set("Content-Type", "application/json")
	if req.Stream || acceptsSSE(r.Header) {
		upstreamReq.Header.Set("Accept", "text/event-stream")
	} else {
		upstreamReq.Header.Set("Accept", "application/json")
	}

	resp, err := p.client.Do(upstreamReq)
	if err != nil {
		writeError(w, http.StatusBadGateway, "upstream_error", err.Error())
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		forwardRawJSON(w, resp.StatusCode, body)
		return
	}

	if isSSEContentType(resp.Header.Get("Content-Type")) || req.Stream || acceptsSSE(r.Header) {
		if err := p.streamChatToResponses(w, resp); err != nil {
			log.Printf("stream transform failed: %v", err)
		}
		return
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		writeError(w, http.StatusBadGateway, "upstream_error", err.Error())
		return
	}

	var chatResp ChatCompletionResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		writeError(w, http.StatusBadGateway, "upstream_error", fmt.Sprintf("invalid upstream JSON: %v", err))
		return
	}

	responsesResp, err := convertChatToResponses(req, chatReq, chatResp)
	if err != nil {
		writeError(w, http.StatusBadGateway, "upstream_error", err.Error())
		return
	}

	writeJSON(w, http.StatusOK, responsesResp)
}

func (p *ProxyServer) handleProxy(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/v1/responses" || r.URL.Path == "/healthz" {
		writeError(w, http.StatusNotFound, "not_found", "not found")
		return
	}
	if !strings.HasPrefix(r.URL.Path, "/v1/") {
		writeError(w, http.StatusNotFound, "not_found", "not found")
		return
	}

	upstreamURL := p.cloneUpstreamURL(r.URL)
	upstreamReq, err := http.NewRequestWithContext(r.Context(), r.Method, upstreamURL.String(), r.Body)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "internal_error", err.Error())
		return
	}
	copyHeaders(upstreamReq.Header, r.Header)

	resp, err := p.client.Do(upstreamReq)
	if err != nil {
		writeError(w, http.StatusBadGateway, "upstream_error", err.Error())
		return
	}
	defer resp.Body.Close()

	copyResponseHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

func (p *ProxyServer) cloneUpstreamURL(incoming *url.URL) *url.URL {
	base := *p.upstream
	base.Path = joinURLPath(base.Path, incoming.Path)
	base.RawQuery = incoming.RawQuery
	return &base
}

func convertResponsesRequest(req ResponsesRequest) (ChatCompletionsRequest, error) {
	if strings.TrimSpace(req.Model) == "" {
		return ChatCompletionsRequest{}, errors.New("model is required")
	}

	messages, err := buildMessages(req)
	if err != nil {
		return ChatCompletionsRequest{}, err
	}

	chatReq := ChatCompletionsRequest{
		Model:            req.Model,
		Messages:         messages,
		Temperature:      req.Temperature,
		TopP:             req.TopP,
		MaxTokens:        req.MaxOutputTokens,
		Metadata:         req.Metadata,
		User:             req.User,
		PresencePenalty:  req.PresencePenalty,
		FrequencyPenalty: req.FrequencyPenalty,
		ToolChoice:       normalizeToolChoice(req.ToolChoice),
		ParallelTools:    req.ParallelToolCalls,
		Stream:           req.Stream,
	}

	for _, tool := range req.Tools {
		if tool.Type != "function" || tool.Function == nil {
			continue
		}
		chatReq.Tools = append(chatReq.Tools, ChatTool{
			Type: "function",
			Function: ChatFunction{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters:  tool.Function.Parameters,
			},
		})
	}

	return chatReq, nil
}

func buildMessages(req ResponsesRequest) ([]ChatMessage, error) {
	messages := make([]ChatMessage, 0, 8)
	if text := strings.TrimSpace(req.Instructions); text != "" {
		messages = append(messages, ChatMessage{Role: "system", Content: text})
	}

	inputMessages, err := normalizeInput(req.Input)
	if err != nil {
		return nil, err
	}
	messages = append(messages, inputMessages...)
	if len(messages) == 0 {
		return nil, errors.New("input or instructions is required")
	}
	return messages, nil
}

func normalizeInput(input any) ([]ChatMessage, error) {
	switch v := input.(type) {
	case nil:
		return nil, nil
	case string:
		if strings.TrimSpace(v) == "" {
			return nil, nil
		}
		return []ChatMessage{{Role: "user", Content: v}}, nil
	case map[string]any:
		msg, err := normalizeInputItem(v)
		if err != nil {
			return nil, err
		}
		if isEmptyMessage(msg) {
			return nil, nil
		}
		return []ChatMessage{msg}, nil
	case []any:
		messages := make([]ChatMessage, 0, len(v))
		for _, item := range v {
			obj, ok := item.(map[string]any)
			if !ok {
				return nil, errors.New("input array items must be objects")
			}
			msg, err := normalizeInputItem(obj)
			if err != nil {
				return nil, err
			}
			if isEmptyMessage(msg) {
				continue
			}
			messages = append(messages, msg)
		}
		return messages, nil
	default:
		return nil, fmt.Errorf("unsupported input type %T", input)
	}
}

func normalizeInputItem(obj map[string]any) (ChatMessage, error) {
	msgType := asString(obj["type"])
	role := asString(obj["role"])
	if role == "" {
		role = defaultInputRole(msgType)
	}
	if role == "" {
		role = "user"
	}

	if msgType == "function_call_output" {
		callID := asString(obj["call_id"])
		output, err := stringifyValue(obj["output"])
		if err != nil {
			return ChatMessage{}, err
		}
		if callID == "" {
			return ChatMessage{Role: "assistant", Content: output}, nil
		}
		return ChatMessage{Role: "tool", ToolCallID: callID, Content: output}, nil
	}

	content, toolCalls, err := normalizeContent(obj)
	if err != nil {
		return ChatMessage{}, err
	}
	msg := ChatMessage{Role: role, Content: content, ToolCalls: toolCalls}
	return msg, nil
}

func defaultInputRole(kind string) string {
	switch kind {
	case "message", "input_text", "input_image", "input_file":
		return "user"
	case "output_text":
		return "assistant"
	default:
		return ""
	}
}

func normalizeContent(obj map[string]any) (any, []ToolCall, error) {
	if text := rawString(obj["text"]); text != "" {
		return text, nil, nil
	}

	content, ok := obj["content"]
	if !ok {
		return "", nil, nil
	}

	switch v := content.(type) {
	case string:
		return v, nil, nil
	case []any:
		parts := make([]map[string]any, 0, len(v))
		toolCalls := make([]ToolCall, 0)
		for _, item := range v {
			part, ok := item.(map[string]any)
			if !ok {
				continue
			}
			chatPart, toolCall, err := normalizeContentPart(part)
			if err != nil {
				return nil, nil, err
			}
			if chatPart != nil {
				parts = append(parts, chatPart)
			}
			if toolCall != nil {
				toolCalls = append(toolCalls, *toolCall)
			}
		}
		if len(parts) == 1 && parts[0]["type"] == "text" {
			return rawString(parts[0]["text"]), toolCalls, nil
		}
		if len(parts) == 0 && len(toolCalls) == 0 {
			return "", nil, nil
		}
		return parts, toolCalls, nil
	default:
		return nil, nil, errors.New("unsupported content field")
	}
}

func normalizeContentPart(part map[string]any) (map[string]any, *ToolCall, error) {
	switch asString(part["type"]) {
	case "input_text", "output_text", "text":
		text := rawString(part["text"])
		if text == "" {
			return nil, nil, nil
		}
		return map[string]any{"type": "text", "text": text}, nil, nil
	case "input_image", "image_url":
		imageURL := extractImageURL(part)
		if imageURL == "" {
			return nil, nil, errors.New("input_image requires image_url")
		}
		imagePart := map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageURL}}
		if detail := rawString(part["detail"]); detail != "" {
			imagePart["image_url"].(map[string]any)["detail"] = detail
		}
		return imagePart, nil, nil
	case "function_call":
		arguments, err := stringifyValue(part["arguments"])
		if err != nil {
			return nil, nil, err
		}
		callID := asString(part["call_id"])
		if callID == "" {
			callID = newID("call")
		}
		return nil, &ToolCall{
			ID:   callID,
			Type: "function",
			Function: ToolCallFunction{
				Name:      asString(part["name"]),
				Arguments: arguments,
			},
		}, nil
	default:
		if text := rawString(part["text"]); text != "" {
			return map[string]any{"type": "text", "text": text}, nil, nil
		}
		return nil, nil, nil
	}
}

func extractImageURL(part map[string]any) string {
	if raw, ok := part["image_url"]; ok {
		switch v := raw.(type) {
		case string:
			return v
		case map[string]any:
			return rawString(v["url"])
		}
	}
	return rawString(part["url"])
}

func convertChatToResponses(req ResponsesRequest, chatReq ChatCompletionsRequest, chatResp ChatCompletionResponse) (ResponsesAPIResponse, error) {
	resp := ResponsesAPIResponse{
		ID:                 coalesce(chatResp.ID, newID("resp")),
		Object:             "response",
		CreatedAt:          chooseTime(chatResp.Created),
		Status:             "completed",
		Model:              coalesce(chatResp.Model, chatReq.Model),
		Output:             make([]ResponseOutput, 0, len(chatResp.Choices)),
		ParallelToolCalls:  chatReq.ParallelTools,
		Store:              req.Store != nil && *req.Store,
		Temperature:        chatReq.Temperature,
		TopP:               chatReq.TopP,
		MaxOutputTokens:    chatReq.MaxTokens,
		PreviousResponseID: strptr(req.PreviousResponseID),
		Text:               ResponseText{Format: ResponseTextFormat{Type: "text"}},
		ToolChoice:         req.ToolChoice,
		Tools:              req.Tools,
		Metadata:           req.Metadata,
		User:               req.User,
	}

	if chatResp.Usage != nil {
		resp.Usage = &ResponsesUsage{
			InputTokens:  chatResp.Usage.PromptTokens,
			OutputTokens: chatResp.Usage.CompletionTokens,
			TotalTokens:  chatResp.Usage.TotalTokens,
		}
	}

	for idx, choice := range chatResp.Choices {
		blocks := buildResponseBlocks(choice.Message)
		if len(blocks) == 0 {
			continue
		}
		resp.Output = append(resp.Output, ResponseOutput{
			ID:      fmt.Sprintf("msg_%d", idx),
			Type:    "message",
			Status:  "completed",
			Role:    defaultAssistantRole(choice.Message.Role),
			Content: blocks,
		})
	}

	if len(resp.Output) == 0 {
		return ResponsesAPIResponse{}, errors.New("upstream returned no assistant content or tool calls")
	}
	return resp, nil
}

func buildResponseBlocks(msg ChatMessage) []ResponseOutputBlock {
	blocks := make([]ResponseOutputBlock, 0, 4)
	blocks = append(blocks, contentToBlocks(msg.Content)...)
	for _, call := range msg.ToolCalls {
		blocks = append(blocks, ResponseOutputBlock{
			Type:      "function_call",
			CallID:    call.ID,
			Name:      call.Function.Name,
			Arguments: call.Function.Arguments,
		})
	}
	if msg.Role == "tool" && strings.TrimSpace(msg.ToolCallID) != "" {
		blocks = append(blocks, ResponseOutputBlock{
			Type:   "function_call_output",
			CallID: msg.ToolCallID,
			Output: msg.Content,
		})
	}
	if refusal := flattenRefusal(msg.Refusal); refusal != "" {
		blocks = append(blocks, ResponseOutputBlock{Type: "refusal", Text: refusal})
	}
	return blocks
}

func contentToBlocks(content any) []ResponseOutputBlock {
	switch v := content.(type) {
	case string:
		if strings.TrimSpace(v) == "" {
			return nil
		}
		return []ResponseOutputBlock{{Type: "output_text", Text: v, Annotations: []any{}}}
	case []any:
		blocks := make([]ResponseOutputBlock, 0, len(v))
		for _, item := range v {
			obj, ok := item.(map[string]any)
			if !ok {
				continue
			}
			blocks = append(blocks, partToBlocks(obj)...)
		}
		return blocks
	case []map[string]any:
		blocks := make([]ResponseOutputBlock, 0, len(v))
		for _, item := range v {
			blocks = append(blocks, partToBlocks(item)...)
		}
		return blocks
	case map[string]any:
		return partToBlocks(v)
	default:
		return nil
	}
}

func partToBlocks(obj map[string]any) []ResponseOutputBlock {
	switch asString(obj["type"]) {
	case "text":
		if text := rawString(obj["text"]); text != "" {
			return []ResponseOutputBlock{{Type: "output_text", Text: text, Annotations: []any{}}}
		}
	case "image_url":
		if imageURL := extractImageURL(obj); imageURL != "" {
			detail := rawString(obj["detail"])
			if raw, ok := obj["image_url"].(map[string]any); ok && detail == "" {
				detail = rawString(raw["detail"])
			}
			return []ResponseOutputBlock{{Type: "output_image", ImageURL: &ResponseImage{URL: imageURL}, Detail: detail}}
		}
	}
	for _, text := range collectTextParts(obj) {
		if strings.TrimSpace(text) != "" {
			return []ResponseOutputBlock{{Type: "output_text", Text: text, Annotations: []any{}}}
		}
	}
	return nil
}

func (p *ProxyServer) streamChatToResponses(w http.ResponseWriter, resp *http.Response) error {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return errors.New("response writer does not support flushing")
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	state := &StreamState{
		ResponseID: newID("resp"),
		OutputID:   newID("msg"),
		ToolCalls:  map[int]*ToolCallState{},
	}
	p.emitResponseCreated(w, flusher, state)

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)
	var lines []string
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if err := p.handleStreamChunk(w, flusher, state, lines); err != nil {
				return err
			}
			lines = lines[:0]
			continue
		}
		lines = append(lines, line)
	}
	if len(lines) > 0 {
		if err := p.handleStreamChunk(w, flusher, state, lines); err != nil {
			return err
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	p.finishStream(w, flusher, state)
	return nil
}

func (p *ProxyServer) handleStreamChunk(w http.ResponseWriter, flusher http.Flusher, state *StreamState, lines []string) error {
	data := collectSSEData(lines)
	if data == "" {
		return nil
	}
	if strings.TrimSpace(data) == "[DONE]" {
		p.finishStream(w, flusher, state)
		return nil
	}

	var payload map[string]any
	decoder := json.NewDecoder(strings.NewReader(data))
	decoder.UseNumber()
	if err := decoder.Decode(&payload); err != nil {
		return nil
	}

	if id := asString(payload["id"]); id != "" {
		state.ResponseID = id
	}
	choices, _ := payload["choices"].([]any)
	for _, item := range choices {
		choice, ok := item.(map[string]any)
		if !ok {
			continue
		}
		delta, _ := choice["delta"].(map[string]any)
		if delta == nil {
			continue
		}
		p.handleDelta(w, flusher, state, delta)
	}
	return nil
}

func (p *ProxyServer) handleDelta(w http.ResponseWriter, flusher http.Flusher, state *StreamState, delta map[string]any) {
	if !state.MessageStarted {
		role := defaultAssistantRole(asString(delta["role"]))
		p.emitMessageAdded(w, flusher, state, role)
	}

	for _, text := range collectTextParts(delta["content"]) {
		p.emitTextDelta(w, flusher, state, text)
	}
	for _, text := range collectTextParts(delta["text"]) {
		p.emitTextDelta(w, flusher, state, text)
	}
	for _, text := range collectTextParts(delta["reasoning"]) {
		p.emitTextDelta(w, flusher, state, text)
	}
	if refusal := flattenRefusal(delta["refusal"]); refusal != "" {
		p.emitTextDelta(w, flusher, state, refusal)
	}

	toolCalls, _ := delta["tool_calls"].([]any)
	for idx, item := range toolCalls {
		call, ok := item.(map[string]any)
		if !ok {
			continue
		}
		fn, _ := call["function"].(map[string]any)
		stateCall := state.toolCall(idx)
		if id := asString(call["id"]); id != "" {
			stateCall.ID = id
		}
		if name := asString(fn["name"]); name != "" {
			stateCall.Name = name
		}
		if !stateCall.Started && stateCall.Name != "" {
			p.emitToolCallAdded(w, flusher, state, idx, stateCall)
		}
		if args := rawString(fn["arguments"]); args != "" {
			if !stateCall.Started && stateCall.Name != "" {
				p.emitToolCallAdded(w, flusher, state, idx, stateCall)
			}
			stateCall.Arguments.WriteString(args)
			emitSSE(w, flusher, "response.function_call_arguments.delta", map[string]any{
				"type":         "response.function_call_arguments.delta",
				"output_index": idx + 1,
				"delta":        args,
			})
		}
	}
}

func (s *StreamState) toolCall(index int) *ToolCallState {
	if call, ok := s.ToolCalls[index]; ok {
		return call
	}
	call := &ToolCallState{}
	s.ToolCalls[index] = call
	return call
}

func (p *ProxyServer) emitResponseCreated(w http.ResponseWriter, flusher http.Flusher, state *StreamState) {
	emitSSE(w, flusher, "response.created", map[string]any{
		"type": "response.created",
		"response": map[string]any{
			"id":     state.ResponseID,
			"object": "response",
			"status": "in_progress",
		},
	})
}

func (p *ProxyServer) emitMessageAdded(w http.ResponseWriter, flusher http.Flusher, state *StreamState, role string) {
	if state.MessageStarted {
		return
	}
	state.MessageStarted = true
	emitSSE(w, flusher, "response.output_item.added", map[string]any{
		"type":         "response.output_item.added",
		"output_index": 0,
		"item": map[string]any{
			"id":      state.OutputID,
			"type":    "message",
			"status":  "in_progress",
			"role":    role,
			"content": []any{},
		},
	})
}

func (p *ProxyServer) emitTextDelta(w http.ResponseWriter, flusher http.Flusher, state *StreamState, text string) {
	if strings.TrimSpace(text) == "" {
		return
	}
	emitSSE(w, flusher, "response.output_text.delta", map[string]any{
		"type":          "response.output_text.delta",
		"output_index":  0,
		"content_index": 0,
		"delta":         text,
	})
}

func (p *ProxyServer) emitToolCallAdded(w http.ResponseWriter, flusher http.Flusher, state *StreamState, idx int, call *ToolCallState) {
	if call.Started {
		return
	}
	call.Started = true
	callID := coalesce(call.ID, newID("call"))
	call.ID = callID
	emitSSE(w, flusher, "response.output_item.added", map[string]any{
		"type":         "response.output_item.added",
		"output_index": idx + 1,
		"item": map[string]any{
			"id":        callID,
			"type":      "function_call",
			"status":    "in_progress",
			"call_id":   callID,
			"name":      call.Name,
			"arguments": "",
		},
	})
}

func (p *ProxyServer) finishStream(w http.ResponseWriter, flusher http.Flusher, state *StreamState) {
	if state.MessageStarted && !state.MessageFinished {
		state.MessageFinished = true
		emitSSE(w, flusher, "response.output_item.done", map[string]any{
			"type":         "response.output_item.done",
			"output_index": 0,
			"item": map[string]any{
				"id":     state.OutputID,
				"type":   "message",
				"status": "completed",
				"role":   "assistant",
			},
		})
	}
	for idx, call := range state.ToolCalls {
		if !call.Started || call.Finished {
			continue
		}
		call.Finished = true
		emitSSE(w, flusher, "response.output_item.done", map[string]any{
			"type":         "response.output_item.done",
			"output_index": idx + 1,
			"item": map[string]any{
				"id":        call.ID,
				"type":      "function_call",
				"status":    "completed",
				"call_id":   call.ID,
				"name":      call.Name,
				"arguments": call.Arguments.String(),
			},
		})
	}
	emitSSE(w, flusher, "response.completed", map[string]any{
		"type": "response.completed",
		"response": map[string]any{
			"id":     state.ResponseID,
			"object": "response",
			"status": "completed",
		},
	})
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func emitSSE(w io.Writer, flusher http.Flusher, event string, payload any) {
	_, _ = io.WriteString(w, "event: "+event+"\n")
	_, _ = io.WriteString(w, "data: "+mustJSON(payload)+"\n\n")
	flusher.Flush()
}

func collectTextParts(v any) []string {
	switch t := v.(type) {
	case nil:
		return nil
	case string:
		if strings.TrimSpace(t) == "" {
			return nil
		}
		return []string{t}
	case map[string]any:
		for _, key := range []string{"text", "content", "delta", "value"} {
			if s := rawString(t[key]); s != "" {
				return []string{s}
			}
		}
		return nil
	case []any:
		parts := make([]string, 0, len(t))
		for _, item := range t {
			parts = append(parts, collectTextParts(item)...)
		}
		return parts
	default:
		return nil
	}
}

func normalizeBaseURL(raw string) (*url.URL, error) {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return nil, err
	}
	if u.Scheme == "" || u.Host == "" {
		return nil, errors.New("scheme and host are required")
	}
	if u.Path == "" {
		u.Path = "/"
	}
	return u, nil
}

func acceptsSSE(header http.Header) bool {
	return strings.Contains(strings.ToLower(header.Get("Accept")), "text/event-stream")
}

func isSSEContentType(contentType string) bool {
	mediaType, _, err := mime.ParseMediaType(contentType)
	if err != nil {
		return strings.Contains(strings.ToLower(contentType), "text/event-stream")
	}
	return mediaType == "text/event-stream"
}

func copyHeaders(dst, src http.Header) {
	for k, values := range src {
		switch http.CanonicalHeaderKey(k) {
		case "Content-Length", "Host", "Accept-Encoding":
			continue
		}
		for _, value := range values {
			dst.Add(k, value)
		}
	}
}

func copyResponseHeaders(dst, src http.Header) {
	for k, values := range src {
		for _, value := range values {
			dst.Add(k, value)
		}
	}
}

func joinURLPath(basePath, requestPath string) string {
	if basePath == "" || basePath == "/" {
		return requestPath
	}
	return path.Join(basePath, requestPath)
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func forwardRawJSON(w http.ResponseWriter, status int, body []byte) {
	if len(body) == 0 {
		writeError(w, status, "upstream_error", http.StatusText(status))
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(body)
}

func writeError(w http.ResponseWriter, status int, errType, message string) {
	writeJSON(w, status, ErrorEnvelope{Error: APIError{Message: message, Type: errType}})
}

func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
	})
}

func collectSSEData(lines []string) string {
	parts := make([]string, 0, len(lines))
	for _, line := range lines {
		if strings.HasPrefix(line, "data:") {
			parts = append(parts, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	return strings.Join(parts, "\n")
}

func normalizeToolChoice(v any) any {
	if m, ok := v.(map[string]any); ok {
		if asString(m["type"]) == "function" {
			if name := asString(m["name"]); name != "" {
				return map[string]any{"type": "function", "function": map[string]any{"name": name}}
			}
		}
	}
	return v
}

func stringifyValue(v any) (string, error) {
	switch t := v.(type) {
	case nil:
		return "", nil
	case string:
		return t, nil
	default:
		buf, err := json.Marshal(t)
		if err != nil {
			return "", err
		}
		return string(buf), nil
	}
}

func flattenRefusal(v any) string {
	switch t := v.(type) {
	case nil:
		return ""
	case string:
		return t
	case []any:
		parts := make([]string, 0, len(t))
		for _, item := range t {
			if s := flattenRefusal(item); strings.TrimSpace(s) != "" {
				parts = append(parts, s)
			}
		}
		return strings.Join(parts, "\n")
	case map[string]any:
		if s := rawString(t["text"]); s != "" {
			return s
		}
		return rawString(t["content"])
	default:
		return ""
	}
}

func chooseTime(v int64) int64 {
	if v > 0 {
		return v
	}
	return time.Now().Unix()
}

func asString(v any) string {
	s, _ := v.(string)
	return strings.TrimSpace(s)
}

func rawString(v any) string {
	s, _ := v.(string)
	return s
}

func isEmptyMessage(msg ChatMessage) bool {
	if len(msg.ToolCalls) > 0 || strings.TrimSpace(msg.ToolCallID) != "" {
		return false
	}
	switch v := msg.Content.(type) {
	case nil:
		return true
	case string:
		return strings.TrimSpace(v) == ""
	case []any:
		return len(v) == 0
	case []map[string]any:
		return len(v) == 0
	default:
		return false
	}
}

func defaultAssistantRole(role string) string {
	if strings.TrimSpace(role) == "" {
		return "assistant"
	}
	return role
}

func strptr(v string) *string {
	if strings.TrimSpace(v) == "" {
		return nil
	}
	return &v
}

func coalesce(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func mustJSON(v any) string {
	buf, err := json.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(buf)
}

func newID(prefix string) string {
	buf := make([]byte, 8)
	if _, err := rand.Read(buf); err != nil {
		return fmt.Sprintf("%s_%d", prefix, time.Now().UnixNano())
	}
	return prefix + "_" + hex.EncodeToString(buf)
}

func init() {
	log.SetOutput(os.Stdout)
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
}
