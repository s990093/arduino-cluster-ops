import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "Micro-CUDA",
  description: "SIMT Architecture on ESP32 Dual-Core System",
  base: "/arduino-cluster-ops/", // Repository name for GitHub Pages
  
  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }]
  ],

  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Guide', link: '/guide/introduction' },
      { text: 'ISA Reference', link: '/guide/isa' },
      { text: 'Paper', link: 'https://github.com/s990093/arduino-cluster-ops/blob/main/docs/paper/main.pdf' }
    ],

    sidebar: [
      {
        text: 'Guide',
        items: [
          { text: 'Introduction', link: '/guide/introduction' },
          { text: 'System Architecture', link: '/guide/architecture' },
          { text: 'Hardware Implementation', link: '/guide/hardware-implementation' },
          { text: 'Interconnect & Synchronization', link: '/guide/interconnect' },
          { text: 'ISA Reference', link: '/guide/isa' },
          { text: 'Software Stack', link: '/guide/software' },
          { text: 'Evaluation', link: '/guide/evaluation' },
          { text: 'Conclusion', link: '/guide/conclusion' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/s990093/arduino-cluster-ops' }
    ],
    
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024 Hung-Wei Machine'
    }
  }
})
