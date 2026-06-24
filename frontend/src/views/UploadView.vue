<template>
  <div class="upload-page">
    <div class="hero">
      <h1>🏸 BadmintonCoach</h1>
      <p>上传羽毛球视频，AI自动分析动作、检测问题、生成教练建议</p>
    </div>

    <div class="card upload-card">
      <div
        class="dropzone"
        :class="{ dragging: isDragging, 'has-file': selectedFile }"
        @dragover.prevent="isDragging = true"
        @dragleave="isDragging = false"
        @drop.prevent="onDrop"
        @click="fileInput?.click()"
      >
        <input
          ref="fileInput"
          type="file"
          accept="video/*"
          style="display: none"
          @change="onFileSelect"
        />
        <div v-if="!selectedFile" class="dropzone-content">
          <div class="icon">📹</div>
          <p>拖拽视频到这里，或点击选择</p>
          <span class="hint">支持 MP4, MOV, AVI, MKV（最大500MB）</span>
        </div>
        <div v-else class="dropzone-content">
          <div class="icon">✅</div>
          <p>{{ selectedFile.name }}</p>
          <span class="hint">{{ formatSize(selectedFile.size) }}</span>
        </div>
      </div>

      <div v-if="store.error" class="error-msg">{{ store.error }}</div>

      <button
        class="btn btn-primary upload-btn"
        :disabled="!selectedFile || store.loading"
        @click="handleUpload"
      >
        {{ store.loading ? '上传中...' : '🚀 上传并分析' }}
      </button>
    </div>

    <div class="features">
      <div class="feature-card card">
        <h3>🦴 骨骼标定</h3>
        <p>RTMPose实时标注17个关键点</p>
      </div>
      <div class="feature-card card">
        <h3>🏸 球体追踪</h3>
        <p>TrackNet热力图追踪羽毛球</p>
      </div>
      <div class="feature-card card">
        <h3>📊 运动参数</h3>
        <p>关节角度、速度、身体倾斜</p>
      </div>
      <div class="feature-card card">
        <h3>🤖 AI教练</h3>
        <p>LLM生成专业教练建议</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAnalysisStore } from '../stores/analysis'

const router = useRouter()
const store = useAnalysisStore()
const fileInput = ref<HTMLInputElement>()
const selectedFile = ref<File | null>(null)
const isDragging = ref(false)

const onFileSelect = (e: Event) => {
  const input = e.target as HTMLInputElement
  if (input.files?.length) {
    selectedFile.value = input.files[0]
  }
}

const onDrop = (e: DragEvent) => {
  isDragging.value = false
  if (e.dataTransfer?.files?.length) {
    selectedFile.value = e.dataTransfer.files[0]
  }
}

const formatSize = (bytes: number) => {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

const handleUpload = async () => {
  if (!selectedFile.value) return
  try {
    const videoId = await store.upload(selectedFile.value)
    await store.start()
    router.push(`/analysis/${videoId}`)
  } catch {}
}
</script>

<style scoped>
.upload-page {
  max-width: 700px;
  margin: 0 auto;
}
.hero {
  text-align: center;
  padding: 40px 0 24px;
}
.hero h1 { font-size: 32px; margin-bottom: 8px; }
.hero p { color: var(--text-secondary); }
.upload-card { margin-bottom: 32px; }
.dropzone {
  border: 2px dashed var(--border);
  border-radius: 12px;
  padding: 48px 24px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
  margin-bottom: 16px;
}
.dropzone:hover, .dropzone.dragging {
  border-color: var(--accent);
  background: rgba(79, 195, 247, 0.05);
}
.dropzone.has-file {
  border-color: var(--success);
  background: rgba(102, 187, 106, 0.05);
}
.icon { font-size: 48px; margin-bottom: 12px; }
.hint { color: var(--text-secondary); font-size: 13px; }
.upload-btn { width: 100%; padding: 14px; font-size: 16px; }
.error-msg {
  color: var(--danger);
  margin-bottom: 12px;
  font-size: 14px;
}
.features {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}
.feature-card h3 { margin-bottom: 8px; }
.feature-card p { color: var(--text-secondary); font-size: 14px; }
</style>
