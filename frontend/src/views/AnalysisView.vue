<template>
  <div class="analysis-page">
    <!-- 加载状态 -->
    <div v-if="!store.result && !store.error" class="loading-card card">
      <h2>🏸 分析中...</h2>
      <div class="progress-bar" style="margin: 16px 0">
        <div class="progress-bar-fill" :style="{ width: (store.status?.progress || 0) * 100 + '%' }"></div>
      </div>
      <p>{{ store.status?.message || '准备中...' }}</p>
      <p class="progress-pct">{{ ((store.status?.progress || 0) * 100).toFixed(0) }}%</p>
    </div>

    <!-- 错误 -->
    <div v-if="store.error" class="error-card card">
      <h2>❌ 分析失败</h2>
      <p>{{ store.error }}</p>
      <button class="btn btn-primary" @click="goBack">返回重试</button>
    </div>

    <!-- 结果 -->
    <div v-if="store.result" class="result-layout">
      <!-- 左侧：视频播放器 + 时间轴 -->
      <div class="left-panel">
        <div class="card video-card">
          <h3>📊 标注视频</h3>
          <video
            ref="videoRef"
            :src="videoUrl"
            controls
            class="video-player"
            @timeupdate="onTimeUpdate"
          ></video>
        </div>

        <!-- 时间轴标记 -->
        <div class="card timeline-card">
          <h3>⏱ 事件时间轴</h3>
          <div class="timeline-bar">
            <div
              v-for="ev in store.events"
              :key="ev.frame_idx"
              class="timeline-marker"
              :class="ev.event_type"
              :style="{ left: (ev.timestamp / duration) * 100 + '%' }"
              :title="ev.description"
              @click="seekTo(ev.timestamp)"
            ></div>
          </div>
          <div class="event-list">
            <div
              v-for="ev in store.events"
              :key="ev.frame_idx"
              class="event-item"
              :class="ev.event_type"
              @click="seekTo(ev.timestamp)"
            >
              <span class="badge" :class="'badge-' + ev.event_type">
                {{ ev.event_type === 'problem' ? '⚠ 问题' : '★ 精彩' }}
              </span>
              <span class="event-desc">{{ ev.description }}</span>
              <span class="event-time">{{ ev.timestamp.toFixed(1) }}s</span>
            </div>
          </div>
        </div>

        <!-- 慢放视频 -->
        <div class="card" v-if="hasSlowmo">
          <h3>🎬 问题集锦（慢放）</h3>
          <video :src="problemsUrl" controls class="video-player"></video>
        </div>
        <div class="card" v-if="hasSlowmo">
          <h3>🌟 精彩瞬间（慢放）</h3>
          <video :src="highlightsUrl" controls class="video-player"></video>
        </div>
      </div>

      <!-- 右侧：指标 + LLM -->
      <div class="right-panel">
        <!-- 基础指标 -->
        <div class="card">
          <h3>📈 分析概览</h3>
          <div class="metrics-grid">
            <div class="metric">
              <div class="metric-value">{{ store.result.result.total_frames }}</div>
              <div class="metric-label">总帧数</div>
            </div>
            <div class="metric">
              <div class="metric-value">{{ store.result.result.fps.toFixed(0) }}</div>
              <div class="metric-label">FPS</div>
            </div>
            <div class="metric">
              <div class="metric-value">{{ store.result.result.duration.toFixed(1) }}s</div>
              <div class="metric-label">时长</div>
            </div>
            <div class="metric">
              <div class="metric-value">{{ store.events.length }}</div>
              <div class="metric-label">事件数</div>
            </div>
          </div>
        </div>

        <!-- 事件统计 -->
        <div class="card">
          <h3>📊 事件统计</h3>
          <div class="stats">
            <div class="stat-row">
              <span class="badge badge-problem">⚠ 问题</span>
              <span>{{ problemCount }} 个</span>
            </div>
            <div class="stat-row">
              <span class="badge badge-highlight">★ 精彩</span>
              <span>{{ highlightCount }} 个</span>
            </div>
          </div>
        </div>

        <!-- LLM教练建议 -->
        <div class="card coaching-card">
          <h3>🤖 AI教练建议</h3>
          <div v-if="store.result.coaching" class="coaching-content">
            <pre>{{ store.result.coaching }}</pre>
          </div>
          <div v-else class="no-coaching">
            <p>未配置LLM API Key，无法生成教练建议</p>
            <router-link to="/settings" class="btn btn-primary" style="margin-top:8px">
              去配置
            </router-link>
          </div>
        </div>

        <!-- LLM对话 -->
        <div class="card chat-card">
          <h3>💬 追问教练</h3>
          <div class="chat-messages" ref="chatBox">
            <div v-for="(msg, i) in chatMessages" :key="i" :class="['chat-msg', msg.role]">
              <div class="msg-content">{{ msg.content }}</div>
            </div>
          </div>
          <div class="chat-input">
            <input
              v-model="chatInput"
              placeholder="输入问题，例如：我的挥拍有什么问题？"
              @keyup.enter="sendChat"
            />
            <button class="btn btn-primary" @click="sendChat" :disabled="chatLoading">
              {{ chatLoading ? '...' : '发送' }}
            </button>
          </div>
        </div>

        <!-- 返回 -->
        <button class="btn btn-danger" style="width:100%;margin-top:16px" @click="goBack">
          分析新视频
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAnalysisStore } from '../stores/analysis'
import { llmChat, getFileUrl } from '../api'

const route = useRoute()
const router = useRouter()
const store = useAnalysisStore()
const videoRef = ref<HTMLVideoElement>()
const chatBox = ref<HTMLElement>()
const chatInput = ref('')
const chatLoading = ref(false)
const currentTime = ref(0)
const chatMessages = ref<{ role: string; content: string }[]>([])

const videoId = route.params.id as string

const videoUrl = computed(() =>
  store.result ? getFileUrl(videoId, 'full_analysis.mp4') : ''
)
const problemsUrl = computed(() => getFileUrl(videoId, 'problems_slowmo.mp4'))
const highlightsUrl = computed(() => getFileUrl(videoId, 'highlights_slowmo.mp4'))
const hasSlowmo = computed(() => store.result?.result?.output_files?.problems_slowmo)
const duration = computed(() => store.result?.result?.duration || 1)
const problemCount = computed(() => store.events.filter(e => e.event_type === 'problem').length)
const highlightCount = computed(() => store.events.filter(e => e.event_type === 'highlight').length)

const onTimeUpdate = () => {
  if (videoRef.value) currentTime.value = videoRef.value.currentTime
}

const seekTo = (time: number) => {
  if (videoRef.value) {
    videoRef.value.currentTime = time
    videoRef.value.play()
  }
}

const sendChat = async () => {
  if (!chatInput.value.trim()) return
  const msg = chatInput.value.trim()
  chatMessages.value.push({ role: 'user', content: msg })
  chatInput.value = ''
  chatLoading.value = true
  await nextTick()
  if (chatBox.value) chatBox.value.scrollTop = chatBox.value.scrollHeight
  try {
    const { reply } = await llmChat(msg, videoId)
    chatMessages.value.push({ role: 'assistant', content: reply })
  } catch {
    chatMessages.value.push({ role: 'assistant', content: '请求失败，请检查LLM配置' })
  }
  chatLoading.value = false
  await nextTick()
  if (chatBox.value) chatBox.value.scrollTop = chatBox.value.scrollHeight
}

const goBack = () => {
  store.reset()
  router.push('/')
}

onMounted(() => {
  if (!store.videoId) {
    store.videoId = videoId
    store.pollStatus()
  }
})
</script>

<style scoped>
.analysis-page { max-width: 1200px; }
.loading-card, .error-card { text-align: center; padding: 48px; }
.progress-pct { font-size: 24px; font-weight: 700; color: var(--accent); margin-top: 8px; }
.result-layout { display: grid; grid-template-columns: 1fr 380px; gap: 20px; }
.left-panel { display: flex; flex-direction: column; gap: 16px; }
.right-panel { display: flex; flex-direction: column; gap: 16px; }
.video-player { width: 100%; border-radius: 8px; margin-top: 12px; }
.timeline-bar { position: relative; height: 24px; background: var(--bg-secondary); border-radius: 4px; margin: 12px 0; }
.timeline-marker {
  position: absolute; top: 0; width: 4px; height: 100%; cursor: pointer; border-radius: 2px;
}
.timeline-marker.problem { background: var(--danger); }
.timeline-marker.highlight { background: var(--warning); }
.event-list { max-height: 200px; overflow-y: auto; }
.event-item {
  display: flex; align-items: center; gap: 8px; padding: 6px 0;
  border-bottom: 1px solid var(--border); cursor: pointer;
}
.event-item:hover { background: rgba(255,255,255,0.03); }
.event-desc { flex: 1; font-size: 13px; }
.event-time { color: var(--text-secondary); font-size: 12px; }
.metrics-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
.metric { text-align: center; }
.metric-value { font-size: 24px; font-weight: 700; color: var(--accent); }
.metric-label { font-size: 12px; color: var(--text-secondary); }
.stats { display: flex; flex-direction: column; gap: 8px; }
.stat-row { display: flex; justify-content: space-between; align-items: center; }
.coaching-content pre {
  white-space: pre-wrap; font-size: 13px; line-height: 1.6;
  background: var(--bg-secondary); padding: 12px; border-radius: 8px;
  max-height: 400px; overflow-y: auto;
}
.no-coaching { text-align: center; color: var(--text-secondary); }
.chat-card { display: flex; flex-direction: column; }
.chat-messages { max-height: 300px; overflow-y: auto; margin: 12px 0; }
.chat-msg { margin-bottom: 8px; }
.chat-msg.user .msg-content {
  background: var(--accent); color: #000; padding: 8px 12px;
  border-radius: 12px 12px 0 12px; display: inline-block; max-width: 80%;
}
.chat-msg.assistant .msg-content {
  background: var(--bg-secondary); padding: 8px 12px;
  border-radius: 12px 12px 12px 0; display: inline-block; max-width: 80%;
}
.chat-input { display: flex; gap: 8px; }
.chat-input input { flex: 1; }
</style>
