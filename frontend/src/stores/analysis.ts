import { defineStore } from 'pinia'
import { ref } from 'vue'
import {
  type AnalysisResult,
  type AnalysisStatus,
  type TimelineEvent,
  getAnalysisResult,
  getAnalysisStatus,
  startAnalysis,
  uploadVideo,
} from '../api'

export const useAnalysisStore = defineStore('analysis', () => {
  const videoId = ref<string | null>(null)
  const status = ref<AnalysisStatus | null>(null)
  const result = ref<AnalysisResult | null>(null)
  const events = ref<TimelineEvent[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)

  const upload = async (file: File) => {
    loading.value = true
    error.value = null
    try {
      const resp = await uploadVideo(file)
      videoId.value = resp.video_id
      return resp.video_id
    } catch (e: any) {
      error.value = e.response?.data?.detail || '上传失败'
      throw e
    } finally {
      loading.value = false
    }
  }

  const start = async () => {
    if (!videoId.value) return
    loading.value = true
    error.value = null
    try {
      await startAnalysis(videoId.value)
      pollStatus()
    } catch (e: any) {
      error.value = e.response?.data?.detail || '启动分析失败'
    } finally {
      loading.value = false
    }
  }

  const pollStatus = async () => {
    if (!videoId.value) return
    const poll = async () => {
      try {
        const s = await getAnalysisStatus(videoId.value!)
        status.value = s
        if (s.status === 'done') {
          await fetchResult()
          return
        }
        if (s.status === 'error') {
          error.value = s.message
          return
        }
        setTimeout(poll, 1000)
      } catch {
        setTimeout(poll, 2000)
      }
    }
    poll()
  }

  const fetchResult = async () => {
    if (!videoId.value) return
    try {
      const r = await getAnalysisResult(videoId.value)
      result.value = r
      events.value = r.result?.events || []
    } catch (e: any) {
      error.value = '获取结果失败'
    }
  }

  const reset = () => {
    videoId.value = null
    status.value = null
    result.value = null
    events.value = []
    error.value = null
  }

  return {
    videoId, status, result, events, loading, error,
    upload, start, pollStatus, fetchResult, reset,
  }
})
