
document.addEventListener('DOMContentLoaded', () => {

    // Dataset upload filename
    const input = document.getElementById('dataset');
    const label = document.getElementById('file-name');

    if (input && label) {
        input.addEventListener('change', () => {
            label.textContent = input.files.length
                ? input.files[0].name
                : 'No file selected';
        });
    }

    // EDA chart rendering
    const chartElements = document.querySelectorAll(
    '.plot-area[data-chart], .chart[data-chart]'
    );

    if (chartElements.length && typeof Plotly === 'undefined') {
        console.error('Plotly library is not loaded.');
    }

    chartElements.forEach((element) => {
        try {
            const chartConfig = JSON.parse(
                element.getAttribute('data-chart')
            );

            if (!chartConfig.data || !chartConfig.data.length) {
                throw new Error('No chart data available.');
            }

            Plotly.newPlot(
                element,
                chartConfig.data,
                {
                    ...chartConfig.layout,
                    autosize: true,
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: {
                        family: 'Inter, sans-serif',
                        color: '#64748b'
                    },
                    margin: {
                        l: 45,
                        r: 20,
                        t: 45,
                        b: 45,
                        ...chartConfig.layout?.margin
                    }
                },
                {
                    responsive: true,
                    displaylogo: false,
                    scrollZoom: true
                }
            ).catch(error => {
                console.error('Chart rendering failed:', error);
            });

        } catch (error) {
            console.error('EDA chart error:', error);
            element.textContent = 'Unable to display this chart.';
        }
    });

});
