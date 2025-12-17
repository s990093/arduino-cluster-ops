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
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'ISA Reference', link: '/guide/isa' },
      { text: 'Paper', link: 'https://github.com/s990093/arduino-cluster-ops/blob/main/docs/paper/main.pdf' }
    ],

    sidebar: [
      {
        text: 'Introduction',
        items: [
          { text: 'What is Micro-CUDA?', link: '/' },
          { text: 'Getting Started', link: '/guide/getting-started' }
        ]
      },
        {
          text: 'Guides',
          items: [
            { text: 'Getting Started', link: '/guide/getting-started' },
            { text: 'Architecture Reference', link: '/guide/architecture' },
            { text: 'ISA Specification', link: '/guide/isa' },
            { text: 'Python SDK', link: '/guide/sdk' },
            { text: 'Benchmarks', link: '/guide/benchmarks' }
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
