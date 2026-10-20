/* Tooltips for the owner's dashboard charts (app/templates/partials/admin/bars.html).
   One tooltip per column, every series in it, the value leading. Built with
   textContent: labels are this site's own, but the habit stays. Opens beside
   the column, never over it. Everything here is also in a table on the page. */
(function () {
  document.querySelectorAll('[data-viz-chart]').forEach(function (box) {
    var tip = box.querySelector('.ai-tip');
    function show(col) {
      var data = JSON.parse(col.getAttribute('data-tip'));
      tip.textContent = '';
      var head = document.createElement('div');
      head.className = 'ai-tip-head';
      var v = document.createElement('strong');
      v.textContent = Number(data.total).toLocaleString() + (data.unit ? ' ' + data.unit : '');
      head.appendChild(v);
      head.appendChild(document.createTextNode(' · ' + data.title));
      tip.appendChild(head);
      data.rows.forEach(function (r) {
        var row = document.createElement('div');
        row.className = 'ai-tip-row';
        var key = document.createElement('span');
        key.className = 'ai-tip-key viz-s' + r.slot;
        var val = document.createElement('strong');
        val.textContent = Number(r.n).toLocaleString();
        row.appendChild(key);
        row.appendChild(val);
        row.appendChild(document.createTextNode(' ' + r.label));
        tip.appendChild(row);
      });
      var c = col.getBoundingClientRect(), b = box.getBoundingClientRect();
      tip.hidden = false;
      var right = c.right - b.left + 6, left = c.left - b.left - tip.offsetWidth - 6;
      tip.style.left = Math.max(0, right + tip.offsetWidth <= b.width ? right : left) + 'px';
      tip.style.top = '.5rem';
      box.querySelectorAll('.ai-col').forEach(function (g) { g.classList.toggle('is-active', g === col); });
    }
    function hide() {
      tip.hidden = true;
      box.querySelectorAll('.ai-col').forEach(function (g) { g.classList.remove('is-active'); });
    }
    box.querySelectorAll('.ai-col').forEach(function (col) {
      col.addEventListener('pointerenter', function () { show(col); });
      col.addEventListener('focus', function () { show(col); });
      col.addEventListener('blur', hide);
    });
    box.addEventListener('pointerleave', hide);
  });
})();
