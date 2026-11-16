
document.addEventListener('DOMContentLoaded', () => {

    const input = document.getElementById('dataset');
    const label = document.getElementById('file-name');

    if (input && label) {

        input.addEventListener('change', () => {

            label.textContent = input.files.length
                ? input.files[0].name
                : 'No file selected';

        });

    }

});
