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
	"sort"
	"strings"
	"sync"
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
	ID                string       `json:"id"`
	Object            string       `json:"object"`
	Created           int64        `json:"created"`
	Model             string       `json:"model"`
	Choices           []ChatChoice `json:"choices"`
	Usage             *Usage       `json:"usage,omitempty"`
	SystemFingerprint string       `json:"system_fingerprint,omitempty"`
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
	PreviousResponseID *string          `json:"previous_response_id"`
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
	FileID      string         `json:"file_id,omitempty"`
	FileURL     string         `json:"file_url,omitempty"`
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
	Param   any    `json:"param,omitempty"`
	Code    any    `json:"code,omitempty"`
}

type StreamState struct {
	mu              sync.Mutex
	responseID      string
	model           string
	created         int64
	started         bool
	outputIndex     int
	outputItemID    string
	textIndex       int
	toolCallBuffers map[int]*ToolCallAccumulator
}

type ToolCallAccumulator struct {
	ID        string
	Name      string
	Arguments strings.Builder
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

type ProxyServer struct {
	client   *http.Client
	upstream *url.URL
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
		if err := p.streamChatToResponses(w, r, resp); err != nil {
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
	if _, err := io.Copy(w, resp.Body); err != nil {
		log.Printf("proxy copy failed: %v", err)
	}
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

	if len(req.Tools) > 0 {
		chatReq.Tools = make([]ChatTool, 0, len(req.Tools))
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
	}

	chatReq.Messages = repairToolMessageOrdering(chatReq.Messages)
	return chatReq, nil
}

func buildMessages(req ResponsesRequest) ([]ChatMessage, error) {
	messages := make([]ChatMessage, 0, 16)
	if text := sanitizeUserVisibleText(req.Instructions); strings.TrimSpace(text) != "" {
		messages = append(messages, ChatMessage{Role: "system", Content: text})
	}

	inputMessages, err := normalizeInput(req.Input)
	if err != nil {
		return nil, err
	}
	messages = append(messages, inputMessages...)
	messages = filterEmptyMessages(messages)

	if len(messages) == 0 {
		return nil, errors.New("input or instructions is required")
	}

	return messages, nil
}

func normalizeInput(input any) ([]ChatMessage, error) {
	if input == nil {
		return nil, nil
	}

	switch v := input.(type) {
	case string:
		if strings.TrimSpace(v) == "" {
			return nil, nil
		}
		return []ChatMessage{{Role: "user", Content: v}}, nil
	case []any:
		return normalizeInputArray(v)
	case map[string]any:
		message, err := normalizeInputItem(v)
		if err != nil {
			return nil, err
		}
		return []ChatMessage{message}, nil
	default:
		return nil, fmt.Errorf("unsupported input type %T", input)
	}
}

func normalizeInputArray(items []any) ([]ChatMessage, error) {
	messages := make([]ChatMessage, 0, len(items))
	for _, item := range items {
		obj, ok := item.(map[string]any)
		if !ok {
			return nil, errors.New("input array items must be objects")
		}
		if shouldIgnoreEnvelope(obj) {
			continue
		}
		message, err := normalizeInputItem(obj)
		if err != nil {
			return nil, err
		}
		if isEmptyChatMessage(message) {
			continue
		}
		messages = append(messages, message)
	}
	return messages, nil
}

func normalizeInputItem(obj map[string]any) (ChatMessage, error) {
	if shouldIgnoreEnvelope(obj) {
		return ChatMessage{}, nil
	}

	msgType := asString(obj["type"])
	role := asString(obj["role"])
	if role == "" {
		role = mapInputRole(msgType)
	}
	if role == "" {
		role = "user"
	}

	switch msgType {
	case "function_call_output":
		callID := asString(obj["call_id"])
		if callID == "" {
			if itemRef, ok := obj["item_reference"].(map[string]any); ok {
				callID = asString(itemRef["call_id"])
			}
		}
		if callID == "" {
			if itemID := asString(obj["item_id"]); itemID != "" {
				callID = itemID
			}
		}
		if callID == "" {
			output := sanitizeStructuredValue(obj["output"])
			serialized, err := stringifyValue(output)
			if err != nil {
				return ChatMessage{}, err
			}
			serialized = sanitizeUserVisibleText(serialized)
			if serialized == "" {
				return ChatMessage{}, nil
			}
			return ChatMessage{Role: role, Content: serialized}, nil
		}
		output := sanitizeStructuredValue(obj["output"])
		serialized, err := stringifyValue(output)
		if err != nil {
			return ChatMessage{}, err
		}
		serialized = sanitizeUserVisibleText(serialized)
		if serialized == "" {
			return ChatMessage{}, nil
		}
		return ChatMessage{Role: "tool", ToolCallID: callID, Content: serialized}, nil
	default:
		content, toolCalls, err := extractInputContent(obj)
		if err != nil {
			return ChatMessage{}, err
		}
		msg := ChatMessage{Role: role, Content: content}
		if len(toolCalls) > 0 {
			msg.ToolCalls = toolCalls
		}
		if isEnvelopeLikeAssistantLeak(msg) {
			return ChatMessage{}, nil
		}
		return msg, nil
	}
}

func mapInputRole(kind string) string {
	switch kind {
	case "message":
		return "user"
	case "input_text", "input_image", "input_file":
		return "user"
	case "output_text":
		return "assistant"
	default:
		return ""
	}
}

func extractInputContent(obj map[string]any) (any, []ToolCall, error) {
	if text := sanitizeUserVisibleText(asString(obj["text"])); text != "" {
		return text, nil, nil
	}

	if content, ok := obj["content"]; ok {
		switch v := content.(type) {
		case string:
			return sanitizeUserVisibleText(v), nil, nil
		case []any:
			parts := make([]map[string]any, 0, len(v))
			toolCalls := make([]ToolCall, 0)
			for _, part := range v {
				partObj, ok := part.(map[string]any)
				if !ok {
					continue
				}
				if shouldIgnoreEnvelope(partObj) {
					continue
				}
				chatPart, call, err := mapInputPart(partObj)
				if err != nil {
					return nil, nil, err
				}
				if chatPart != nil {
					parts = append(parts, chatPart)
				}
				if call != nil {
					toolCalls = append(toolCalls, *call)
				}
			}
			if len(parts) == 0 && len(toolCalls) == 0 {
				return "", nil, nil
			}
			if len(parts) == 1 && parts[0]["type"] == "text" {
				if text, _ := parts[0]["text"].(string); text != "" {
					return sanitizeUserVisibleText(text), toolCalls, nil
				}
			}
			return parts, toolCalls, nil
		default:
			return nil, nil, errors.New("unsupported content field")
		}
	}
	return "", nil, nil
}

func mapInputPart(part map[string]any) (map[string]any, *ToolCall, error) {
	if shouldIgnoreEnvelope(part) {
		return nil, nil, nil
	}
	partType := asString(part["type"])
	switch partType {
	case "input_text", "output_text", "text":
		text := sanitizeUserVisibleText(asString(part["text"]))
		if text == "" {
			return nil, nil, nil
		}
		return map[string]any{"type": "text", "text": text}, nil, nil
	case "input_image", "image_url":
		imageURL := ""
		if raw, ok := part["image_url"]; ok {
			switch img := raw.(type) {
			case string:
				imageURL = img
			case map[string]any:
				imageURL = asString(img["url"])
			}
		}
		if imageURL == "" {
			imageURL = asString(part["url"])
		}
		if imageURL == "" {
			return nil, nil, errors.New("input_image requires image_url")
		}
		imageURL = sanitizeUserVisibleText(imageURL)
		if imageURL == "" {
			return nil, nil, nil
		}
		imagePart := map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageURL}}
		if detail := sanitizeUserVisibleText(asString(part["detail"])); detail != "" {
			imagePart["image_url"].(map[string]any)["detail"] = detail
		}
		return imagePart, nil, nil
	case "input_file":
		fileID := sanitizeUserVisibleText(asString(part["file_id"]))
		if fileID != "" {
			return map[string]any{"type": "text", "text": "[file_id:" + fileID + "]"}, nil, nil
		}
		if ref := sanitizeUserVisibleText(asString(part["file_url"])); ref != "" {
			return map[string]any{"type": "text", "text": "[file_url:" + ref + "]"}, nil, nil
		}
		if name := sanitizeUserVisibleText(asString(part["filename"])); name != "" {
			return map[string]any{"type": "text", "text": "[filename:" + name + "]"}, nil, nil
		}
		return nil, nil, nil
	case "function_call":
		callID := sanitizeUserVisibleText(asString(part["call_id"]))
		if callID == "" {
			callID = newID("call")
		}
		arguments, err := stringifyValue(sanitizeStructuredValue(part["arguments"]))
		if err != nil {
			return nil, nil, err
		}
		return nil, &ToolCall{
			ID:   callID,
			Type: "function",
			Function: ToolCallFunction{
				Name:      sanitizeUserVisibleText(asString(part["name"])),
				Arguments: arguments,
			},
		}, nil
	default:
		if text := sanitizeUserVisibleText(asString(part["text"])); text != "" {
			return map[string]any{"type": "text", "text": text}, nil, nil
		}
		return nil, nil, nil
	}
}

func convertChatToResponses(req ResponsesRequest, chatReq ChatCompletionsRequest, chatResp ChatCompletionResponse) (ResponsesAPIResponse, error) {
	resp := ResponsesAPIResponse{
		ID:                 coalesce(chatResp.ID, newID("resp")),
		Object:             "response",
		CreatedAt:          chooseTime(chatResp.Created),
		Status:             "completed",
		Model:              coalesce(chatResp.Model, chatReq.Model),
		ParallelToolCalls:  chatReq.ParallelTools,
		Store:              req.Store != nil && *req.Store,
		Temperature:        chatReq.Temperature,
		TopP:               chatReq.TopP,
		MaxOutputTokens:    chatReq.MaxTokens,
		PreviousResponseID: strptrIfNotEmpty(req.PreviousResponseID),
		Text:               ResponseText{Format: ResponseTextFormat{Type: "text"}},
		ToolChoice:         req.ToolChoice,
		Metadata:           req.Metadata,
		User:               req.User,
	}

	if len(req.Tools) > 0 {
		resp.Tools = req.Tools
	}
	if resp.Metadata == nil && chatResp.Object != "" {
		resp.Metadata = map[string]any{"upstream_object": chatResp.Object}
	}

	if chatResp.Usage != nil {
		resp.Usage = &ResponsesUsage{
			InputTokens:  chatResp.Usage.PromptTokens,
			OutputTokens: chatResp.Usage.CompletionTokens,
			TotalTokens:  chatResp.Usage.TotalTokens,
		}
	}

	for idx, choice := range chatResp.Choices {
		outputID := fmt.Sprintf("msg_%d", idx)
		role := defaultRole(choice.Message.Role)
		blocks := make([]ResponseOutputBlock, 0, 4)

		for _, block := range chatContentToResponseBlocks(choice.Message.Content) {
			blocks = append(blocks, block)
		}
		if role == "tool" {
			blocks = append(blocks, ResponseOutputBlock{
				Type:   "function_call_output",
				CallID: choice.Message.ToolCallID,
				Output: choice.Message.Content,
			})
		}
		for _, call := range choice.Message.ToolCalls {
			blocks = append(blocks, ResponseOutputBlock{
				Type:      "function_call",
				CallID:    call.ID,
				Name:      call.Function.Name,
				Arguments: call.Function.Arguments,
			})
		}
		if refusalText := flattenRefusal(choice.Message.Refusal); refusalText != "" {
			blocks = append(blocks, ResponseOutputBlock{Type: "refusal", Text: refusalText})
		}
		if len(blocks) == 0 {
			continue
		}
		resp.Output = append(resp.Output, ResponseOutput{
			ID:      outputID,
			Type:    "message",
			Status:  "completed",
			Role:    role,
			Content: blocks,
		})
	}

	if len(resp.Output) == 0 {
		return ResponsesAPIResponse{}, errors.New("upstream returned no assistant content or tool calls")
	}

	return resp, nil
}

func chatContentToResponseBlocks(content any) []ResponseOutputBlock {
	switch v := content.(type) {
	case string:
		return []ResponseOutputBlock{{Type: "output_text", Text: v, Annotations: []any{}}}
	case []any:
		blocks := make([]ResponseOutputBlock, 0, len(v))
		for _, item := range v {
			obj, ok := item.(map[string]any)
			if !ok {
				continue
			}
			switch asString(obj["type"]) {
			case "text":
				blocks = append(blocks, ResponseOutputBlock{Type: "output_text", Text: asString(obj["text"]), Annotations: []any{}})
			case "image_url":
				imageURL := ""
				detail := ""
				if raw, ok := obj["image_url"]; ok {
					switch img := raw.(type) {
					case string:
						imageURL = img
					case map[string]any:
						imageURL = asString(img["url"])
						detail = asString(img["detail"])
					}
				}
				blocks = append(blocks, ResponseOutputBlock{Type: "output_image", ImageURL: &ResponseImage{URL: imageURL}, Detail: detail})
			case "input_audio":
				if text := asString(obj["transcript"]); text != "" {
					blocks = append(blocks, ResponseOutputBlock{Type: "output_text", Text: text, Annotations: []any{}})
				}
			}
		}
		return blocks
	default:
		return nil
	}
}

func (p *ProxyServer) streamChatToResponses(w http.ResponseWriter, r *http.Request, resp *http.Response) error {
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

	state := &StreamState{toolCallBuffers: map[int]*ToolCallAccumulator{}}
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)
	var eventLines []string

	emitSSE(w, flusher, []byte("event: response.created\n"+"data: "+mustJSON(map[string]any{
		"type":     "response.created",
		"response": map[string]any{"id": newID("resp"), "object": "response", "status": "in_progress"},
	})+"\n\n"))

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if err := p.processSSEEvent(w, flusher, state, eventLines); err != nil {
				return err
			}
			eventLines = eventLines[:0]
			continue
		}
		eventLines = append(eventLines, line)
	}
	if len(eventLines) > 0 {
		if err := p.processSSEEvent(w, flusher, state, eventLines); err != nil {
			return err
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return nil
}

func (p *ProxyServer) processSSEEvent(w http.ResponseWriter, flusher http.Flusher, state *StreamState, lines []string) error {
	if len(lines) == 0 {
		return nil
	}
	data := collectSSEData(lines)
	if data == "" {
		return nil
	}
	if looksLikeInternalToolTranscript(data) {
		return nil
	}
	if strings.TrimSpace(data) == "[DONE]" {
		finish := map[string]any{"type": "response.completed", "response": map[string]any{"id": state.ensureResponseID(), "object": "response", "status": "completed"}}
		emitSSE(w, flusher, []byte("event: response.completed\n"+"data: "+mustJSON(finish)+"\n\n"))
		emitSSE(w, flusher, []byte("data: [DONE]\n\n"))
		return nil
	}

	var payload map[string]any
	decoder := json.NewDecoder(strings.NewReader(data))
	decoder.UseNumber()
	if err := decoder.Decode(&payload); err != nil {
		return nil
	}

	state.observeChunk(payload)
	choices, _ := payload["choices"].([]any)
	for _, choiceItem := range choices {
		choice, ok := choiceItem.(map[string]any)
		if !ok {
			continue
		}
		delta, _ := choice["delta"].(map[string]any)
		if delta == nil {
			continue
		}

		if role := asString(delta["role"]); role != "" {
			messageCreated := map[string]any{
				"type":         "response.output_item.added",
				"output_index": state.outputIndex,
				"item": map[string]any{
					"id":      state.ensureOutputItemID(),
					"type":    "message",
					"status":  "in_progress",
					"role":    role,
					"content": []any{},
				},
			}
			emitSSE(w, flusher, []byte("event: response.output_item.added\n"+"data: "+mustJSON(messageCreated)+"\n\n"))
		}

		for _, part := range streamContentParts(delta["content"]) {
			textEvent := map[string]any{
				"type":          "response.output_text.delta",
				"output_index":  state.outputIndex,
				"content_index": state.textIndex,
				"delta":         part,
			}
			emitSSE(w, flusher, []byte("event: response.output_text.delta\n"+"data: "+mustJSON(textEvent)+"\n\n"))
		}
		if refusal := flattenRefusal(delta["refusal"]); refusal != "" {
			refusalEvent := map[string]any{
				"type":          "response.refusal.delta",
				"output_index":  state.outputIndex,
				"content_index": state.textIndex,
				"delta":         refusal,
			}
			emitSSE(w, flusher, []byte("event: response.refusal.delta\n"+"data: "+mustJSON(refusalEvent)+"\n\n"))
		}

		toolCalls, _ := delta["tool_calls"].([]any)
		for idx, tc := range toolCalls {
			call, ok := tc.(map[string]any)
			if !ok {
				continue
			}
			fn, _ := call["function"].(map[string]any)
			acc := state.toolAccumulator(idx)
			if id := asString(call["id"]); id != "" {
				acc.ID = id
			}
			if name := asString(fn["name"]); name != "" {
				acc.Name = name
				added := map[string]any{
					"type":         "response.output_item.added",
					"output_index": state.outputIndex + idx + 1,
					"item": map[string]any{
						"id":        coalesce(acc.ID, newID("fc")),
						"type":      "function_call",
						"status":    "in_progress",
						"call_id":   coalesce(acc.ID, newID("call")),
						"name":      acc.Name,
						"arguments": "",
					},
				}
				emitSSE(w, flusher, []byte("event: response.output_item.added\n"+"data: "+mustJSON(added)+"\n\n"))
			}
			if args := asString(fn["arguments"]); args != "" {
				acc.Arguments.WriteString(args)
				deltaEvent := map[string]any{
					"type":         "response.function_call_arguments.delta",
					"output_index": state.outputIndex + idx + 1,
					"delta":        args,
				}
				emitSSE(w, flusher, []byte("event: response.function_call_arguments.delta\n"+"data: "+mustJSON(deltaEvent)+"\n\n"))
			}
		}

		if finishReason := asString(choice["finish_reason"]); finishReason != "" {
			for idx, acc := range state.snapshotToolCalls() {
				doneEvent := map[string]any{
					"type":         "response.output_item.done",
					"output_index": state.outputIndex + idx + 1,
					"item": map[string]any{
						"id":        coalesce(acc.ID, newID("fc")),
						"type":      "function_call",
						"status":    "completed",
						"call_id":   coalesce(acc.ID, newID("call")),
						"name":      acc.Name,
						"arguments": acc.Arguments.String(),
					},
				}
				emitSSE(w, flusher, []byte("event: response.output_item.done\n"+"data: "+mustJSON(doneEvent)+"\n\n"))
			}
			messageDone := map[string]any{
				"type":         "response.output_item.done",
				"output_index": state.outputIndex,
				"item":         map[string]any{"id": state.ensureOutputItemID(), "type": "message", "status": "completed", "role": "assistant"},
			}
			emitSSE(w, flusher, []byte("event: response.output_item.done\n"+"data: "+mustJSON(messageDone)+"\n\n"))
			state.finish(finishReason)
		}
	}
	return nil
}

func streamContentParts(content any) []string {
	switch v := content.(type) {
	case string:
		if v == "" {
			return nil
		}
		return []string{v}
	case []any:
		parts := make([]string, 0, len(v))
		for _, item := range v {
			obj, ok := item.(map[string]any)
			if !ok {
				continue
			}
			text := asString(obj["text"])
			if text != "" {
				parts = append(parts, text)
			}
		}
		return parts
	default:
		return nil
	}
}

func (s *StreamState) observeChunk(payload map[string]any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if id := asString(payload["id"]); id != "" && s.responseID == "" {
		s.responseID = id
	}
	if model := asString(payload["model"]); model != "" && s.model == "" {
		s.model = model
	}
	if s.created == 0 {
		s.created = chooseTime(numberToInt64(payload["created"]))
	}
	if !s.started {
		s.started = true
		s.outputItemID = newID("msg")
	}
}

func (s *StreamState) ensureResponseID() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.responseID == "" {
		s.responseID = newID("resp")
	}
	return s.responseID
}

func (s *StreamState) ensureOutputItemID() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.outputItemID == "" {
		s.outputItemID = newID("msg")
	}
	return s.outputItemID
}

func (s *StreamState) toolAccumulator(index int) *ToolCallAccumulator {
	s.mu.Lock()
	defer s.mu.Unlock()
	acc, ok := s.toolCallBuffers[index]
	if !ok {
		acc = &ToolCallAccumulator{}
		s.toolCallBuffers[index] = acc
	}
	return acc
}

func (s *StreamState) snapshotToolCalls() map[int]ToolCallAccumulator {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make(map[int]ToolCallAccumulator, len(s.toolCallBuffers))
	for idx, acc := range s.toolCallBuffers {
		copied := ToolCallAccumulator{ID: acc.ID, Name: acc.Name}
		copied.Arguments.WriteString(acc.Arguments.String())
		out[idx] = copied
	}
	return out
}

func (s *StreamState) finish(_ string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.textIndex++
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
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		log.Printf("write json failed: %v", err)
	}
}

func forwardRawJSON(w http.ResponseWriter, status int, body []byte) {
	if len(body) == 0 {
		writeError(w, status, "upstream_error", http.StatusText(status))
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if _, err := w.Write(body); err != nil {
		log.Printf("write response failed: %v", err)
	}
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

func emitSSE(w io.Writer, flusher http.Flusher, payload []byte) {
	_, _ = w.Write(payload)
	flusher.Flush()
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
		return strings.TrimSpace(t)
	case []any:
		parts := make([]string, 0, len(t))
		for _, item := range t {
			parts = append(parts, flattenRefusal(item))
		}
		return strings.TrimSpace(strings.Join(parts, "\n"))
	case map[string]any:
		if text := asString(t["text"]); text != "" {
			return text
		}
		if text := asString(t["content"]); text != "" {
			return text
		}
		return ""
	default:
		return strings.TrimSpace(fmt.Sprintf("%v", t))
	}
}

func numberToInt64(v any) int64 {
	switch n := v.(type) {
	case json.Number:
		i, _ := n.Int64()
		return i
	case float64:
		return int64(n)
	case int64:
		return n
	case int:
		return int64(n)
	default:
		return 0
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

func filterEmptyMessages(messages []ChatMessage) []ChatMessage {
	filtered := make([]ChatMessage, 0, len(messages))
	for _, msg := range messages {
		if isEmptyChatMessage(msg) {
			continue
		}
		filtered = append(filtered, msg)
	}
	return filtered
}

func isEmptyChatMessage(msg ChatMessage) bool {
	if len(msg.ToolCalls) > 0 {
		return false
	}
	if strings.TrimSpace(msg.ToolCallID) != "" {
		return false
	}
	switch v := msg.Content.(type) {
	case nil:
		return true
	case string:
		return strings.TrimSpace(v) == ""
	case []map[string]any:
		return len(v) == 0
	case []any:
		return len(v) == 0
	default:
		return false
	}
}

func sanitizeUserVisibleText(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	if looksLikeInternalToolTranscript(s) {
		return ""
	}
	return s
}

func looksLikeInternalToolTranscript(s string) bool {
	lower := strings.ToLower(strings.TrimSpace(s))
	if lower == "" {
		return false
	}
	patterns := []string{
		"<tool_response>",
		"the tool execution was interrupted before it could be completed",
		"you did not use a tool in your previous response",
		"this is an automated message",
		"the language model did not provide any assistant messages",
		"model provided text/reasoning but did not call any required tools",
		"模型提供了文本/推理，但未调用任何必需的工具",
		"file: readme.md",
		"file: main.go",
		"random text follows",
	}
	for _, pattern := range patterns {
		if strings.Contains(lower, pattern) {
			return true
		}
	}
	if strings.HasPrefix(lower, "file: ") && strings.Contains(lower, "| #") {
		return true
	}
	return false
}

func shouldIgnoreEnvelope(obj map[string]any) bool {
	if obj == nil {
		return false
	}
	if toolResp, ok := obj["tool_response"]; ok && toolResp != nil {
		return true
	}
	if role := asString(obj["role"]); role == "tool" && asString(obj["type"]) == "message" {
		if text := sanitizeUserVisibleText(asString(obj["text"])); text == "" && asString(obj["call_id"]) == "" {
			return true
		}
	}
	if typ := asString(obj["type"]); typ == "tool_response" || typ == "tool_result" || typ == "tool_feedback" {
		return true
	}
	for _, key := range []string{"status", "feedback", "guidance", "artifact", "reminder"} {
		if _, ok := obj[key]; ok && strings.HasPrefix(asString(obj["type"]), "tool") {
			return true
		}
	}
	if text := asString(obj["text"]); text != "" && looksLikeInternalToolTranscript(text) {
		return true
	}
	if content, ok := obj["content"].(string); ok && looksLikeInternalToolTranscript(content) {
		return true
	}
	return false
}

func sanitizeStructuredValue(v any) any {
	switch t := v.(type) {
	case string:
		return sanitizeUserVisibleText(t)
	case []any:
		out := make([]any, 0, len(t))
		for _, item := range t {
			sanitized := sanitizeStructuredValue(item)
			if str, ok := sanitized.(string); ok && strings.TrimSpace(str) == "" {
				continue
			}
			if m, ok := sanitized.(map[string]any); ok && len(m) == 0 {
				continue
			}
			out = append(out, sanitized)
		}
		return out
	case map[string]any:
		if shouldIgnoreEnvelope(t) {
			return map[string]any{}
		}
		out := make(map[string]any, len(t))
		for k, item := range t {
			sanitized := sanitizeStructuredValue(item)
			if str, ok := sanitized.(string); ok && strings.TrimSpace(str) == "" {
				continue
			}
			if m, ok := sanitized.(map[string]any); ok && len(m) == 0 {
				continue
			}
			out[k] = sanitized
		}
		return out
	default:
		return v
	}
}

func isEnvelopeLikeAssistantLeak(msg ChatMessage) bool {
	if len(msg.ToolCalls) > 0 || strings.TrimSpace(msg.ToolCallID) != "" {
		return false
	}
	switch v := msg.Content.(type) {
	case string:
		return looksLikeInternalToolTranscript(v)
	default:
		return false
	}
}

func strptrIfNotEmpty(v string) *string {
	v = strings.TrimSpace(v)
	if v == "" {
		return nil
	}
	return &v
}

func defaultRole(role string) string {
	if role == "" {
		return "assistant"
	}
	return role
}

func repairToolMessageOrdering(messages []ChatMessage) []ChatMessage {
	if len(messages) == 0 {
		return messages
	}

	pendingToolCalls := map[string]ToolCall{}
	ordered := make([]ChatMessage, 0, len(messages))
	orphanTools := make([]ChatMessage, 0)

	for _, msg := range messages {
		if len(msg.ToolCalls) > 0 {
			for _, call := range msg.ToolCalls {
				if strings.TrimSpace(call.ID) != "" {
					pendingToolCalls[call.ID] = call
				}
			}
			ordered = append(ordered, msg)
			continue
		}

		if msg.Role == "tool" {
			if strings.TrimSpace(msg.ToolCallID) == "" {
				msg.Role = "assistant"
				ordered = append(ordered, msg)
				continue
			}
			if _, ok := pendingToolCalls[msg.ToolCallID]; ok {
				ordered = append(ordered, msg)
				delete(pendingToolCalls, msg.ToolCallID)
				continue
			}
			orphanTools = append(orphanTools, ChatMessage{Role: "assistant", Content: msg.Content})
			continue
		}

		ordered = append(ordered, msg)
	}

	ordered = append(ordered, orphanTools...)
	return ordered
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
	sort.Strings([]string{})
}
