<template>
  <div id="app">
    <nav class="navbar">
      <router-link to="/" class="logo">🏸 BadmintonCoach</router-link>
      <div class="nav-links">
        <router-link to="/">上传</router-link>
        <router-link to="/settings">设置</router-link>
      </div>
    </nav>
    <main class="main-content">
      <router-view />
    </main>
  </div>
</template>

<script setup lang="ts">
</script>

<style scoped>
.navbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  height: 56px;
  background: #1a1a2e;
  border-bottom: 1px solid #333;
}
.logo {
  font-size: 18px;
  font-weight: 700;
  color: #fff;
  text-decoration: none;
}
.nav-links {
  display: flex;
  gap: 20px;
}
.nav-links a {
  color: #aaa;
  text-decoration: none;
  font-size: 14px;
}
.nav-links a.router-link-active {
  color: #4fc3f7;
}
.main-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px;
}
</style>
