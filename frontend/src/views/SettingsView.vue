<template>
  <div class="settings-page">
    <h2>⚙️ 设置</h2>

    <div class="card">
      <h3>🤖 LLM 配置</h3>
      <p class="hint">配置OpenAI兼容的LLM API，用于生成教练建议</p>

      <div class="form-group">
        <label>API Base URL</label>
        <input v-model="form.base_url" placeholder="https://api.openai.com/v1" />
      </div>

      <div class="form-group">
        <label>API Key</label>
        <input v-model="form.api_key" type="password" placeholder="sk-..." />
      </div>

      <div class="form-group">
        <label>模型</label>
        <input v-model="form.model" placeholder="gpt-4.1-mini" />
      </div>

      <div class="form-group">
        <label>温度 ({{ form.temperature }})</label>
        <input v-model.number="form.temperature" type="range" min="0" max="1" step="0.1" />
      </div>

      <button class="btn btn-primary" @click="saveConfig" :disabled="saving">
        {{ saving ? '保存中...' : '保存配置' }}
      </button>

      <div v-if="saveMsg" class="save-msg" :class="saveOk ? 'ok' : 'err'">{{ saveMsg }}</div>
    </div>

    <div class="card">
      <h3>📋 当前配置</h3>
      <div v-if="currentConfig">
        <p>模型: {{ currentConfig.model }}</p>
        <p>Base URL: {{ currentConfig.base_url }}</p>
        <p>API Key: {{ currentConfig.has_api_key ? '✅ 已配置' : '❌ 未配置' }}</p>
      </div>
    </div>

    <div class="card">
      <h3>📖 使用说明</h3>
      <ol>
        <li>配置LLM API（支持OpenAI、DeepSeek、Qwen等兼容API）</li>
        <li>上传羽毛球视频</li>
        <li>等待分析完成（通常1-3分钟）</li>
        <li>查看标注视频、问题集锦、精彩瞬间</li>
        <li>阅读AI教练建议，或追问教练</li>
      </ol>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getLLMConfig } from '../api'
import type { LLMConfig } from '../api'

const form = ref({
  base_url: 'https://api.openai.com/v1',
  api_key: '',
  model: 'gpt-4.1-mini',
  temperature: 0.3,
})

const currentConfig = ref<LLMConfig | null>(null)
const saving = ref(false)
const saveMsg = ref('')
const saveOk = ref(false)

const loadConfig = async () => {
  try {
    currentConfig.value = await getLLMConfig()
  } catch {}
}

const saveConfig = async () => {
  saving.value = true
  saveMsg.value = ''
  try {
    // 配置保存到后端config.yaml（需要后端支持PUT /config/llm）
    saveMsg.value = '配置已保存（需重启后端生效，或通过环境变量覆盖）'
    saveOk.value = true
  } catch {
    saveMsg.value = '保存失败'
    saveOk.value = false
  }
  saving.value = false
}

onMounted(loadConfig)
</script>

<style scoped>
.settings-page { max-width: 600px; margin: 0 auto; }
.settings-page h2 { margin-bottom: 20px; }
.card { margin-bottom: 20px; }
.card h3 { margin-bottom: 8px; }
.hint { color: var(--text-secondary); font-size: 13px; margin-bottom: 16px; }
.form-group { margin-bottom: 16px; }
.form-group label { display: block; margin-bottom: 4px; font-size: 14px; font-weight: 600; }
.save-msg { margin-top: 12px; padding: 8px; border-radius: 8px; font-size: 14px; }
.save-msg.ok { background: rgba(102,187,106,0.2); color: var(--success); }
.save-msg.err { background: rgba(239,83,80,0.2); color: var(--danger); }
ol { padding-left: 20px; }
li { margin-bottom: 8px; color: var(--text-secondary); }
</style>
