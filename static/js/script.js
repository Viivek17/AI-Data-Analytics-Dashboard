
document.addEventListener('DOMContentLoaded', function () {

    const chartElements = document.querySelectorAll('.chart[data-chart]');

    chartElements.forEach(function (element) {

        try {

            const figure = JSON.parse(element.dataset.chart);

            if (window.Plotly && figure) {

                Plotly.newPlot(
                    element,
                    figure.data,
                    figure.layout,
                    {
                        responsive: true,
                        displaylogo: false
                    }
                );

            }

        } catch (error) {

            console.error(
                'Prediction chart rendering error:',
                element.id,
                error
            );

        }

    });

});
