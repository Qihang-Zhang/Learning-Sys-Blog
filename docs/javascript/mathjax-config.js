window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
    ignoreHtmlClass: 'tex2jax_ignore',
    processHtmlClass: 'tex2jax_process'
  },
  startup: {
    pageReady: () => {
      return MathJax.startup.defaultPageReady().then(() => {
        document.querySelectorAll("script[type='math/tex']").forEach((node) => {
          const math = MathJax.tex2chtml(node.textContent, { display: false });
          node.parentNode.replaceChild(math, node);
        });
      });
    }
  }
};


