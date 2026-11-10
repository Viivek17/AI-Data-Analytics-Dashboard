document.addEventListener('DOMContentLoaded', () => {
  const toggle = document.getElementById('menuToggle');
  const sidebar = document.getElementById('sidebar');
  if (toggle && sidebar) toggle.addEventListener('click', () => sidebar.classList.toggle('open'));

  const input = document.getElementById('datasetFile');
  const selected = document.getElementById('selectedFile');
  const zone = document.getElementById('dropzone');
  if (input && selected) {
    input.addEventListener('change', () => {
      selected.textContent = input.files.length ? `${input.files[0].name} · ${(input.files[0].size / 1024).toFixed(1)} KB` : 'No file selected';
    });
  }
  if (zone && input) {
    ['dragenter', 'dragover'].forEach(evt => zone.addEventListener(evt, e => { e.preventDefault(); zone.classList.add('dragging'); }));
    ['dragleave', 'drop'].forEach(evt => zone.addEventListener(evt, e => { e.preventDefault(); zone.classList.remove('dragging'); }));
    zone.addEventListener('drop', e => {
      if (e.dataTransfer.files.length) {
        input.files = e.dataTransfer.files;
        input.dispatchEvent(new Event('change'));
      }
    });
  }
  const search = document.getElementById('tableSearch');
  const table = document.getElementById('dataTable');
  if (search && table) search.addEventListener('input', () => {
    const term = search.value.toLowerCase();
    table.querySelectorAll('tbody tr').forEach(row => row.style.display = row.textContent.toLowerCase().includes(term) ? '' : 'none');
  });
});

