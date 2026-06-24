import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'upload',
    component: () => import('./views/UploadView.vue'),
  },
  {
    path: '/analysis/:id',
    name: 'analysis',
    component: () => import('./views/AnalysisView.vue'),
  },
  {
    path: '/settings',
    name: 'settings',
    component: () => import('./views/SettingsView.vue'),
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
