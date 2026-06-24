import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 300000, // 5min for large uploads
})

export interface UploadResponse {
  video_id: string
  filename: string
  size: number
  path: string
}

export interface AnalysisStatus {
  video_id: string
  status: string
  progress: number
  message: string
}

export interface TimelineEvent {
  frame_idx: number
  timestamp: number
  event_type: string
  sub_type: string
  severity: string
  description: string
  improvement: string
  start_frame: number
  end_frame: number
  score: number
}

export interface AnalysisResult {
  result: {
    video_id: string
    total_frames: number
    fps: number
    duration: number
    events: TimelineEvent[]
    output_files: Record<string, string>
  }
  coaching: string | null
}

export interface LLMConfig {
  base_url: string
  model: string
  has_api_key: boolean
}

export const uploadVideo = async (file: File): Promise<UploadResponse> => {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export const startAnalysis = async (videoId: string) => {
  const { data } = await api.post(`/analysis/${videoId}`)
  return data
}

export const getAnalysisStatus = async (videoId: string): Promise<AnalysisStatus> => {
  const { data } = await api.get(`/analysis/${videoId}/status`)
  return data
}

export const getAnalysisResult = async (videoId: string): Promise<AnalysisResult> => {
  const { data } = await api.get(`/analysis/${videoId}/result`)
  return data
}

export const getEvents = async (videoId: string): Promise<{ events: TimelineEvent[] }> => {
  const { data } = await api.get(`/analysis/${videoId}/events`)
  return data
}

export const llmChat = async (message: string, videoId?: string): Promise<{ reply: string }> => {
  const { data } = await api.post('/llm/chat', { message, video_id: videoId })
  return data
}

export const getLLMConfig = async (): Promise<LLMConfig> => {
  const { data } = await api.get('/config/llm')
  return data
}

export const getFileUrl = (videoId: string, filename: string) => {
  return `/api/files/${videoId}/${filename}`
}

export default api
