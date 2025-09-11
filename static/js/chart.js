// Chart initialization and management
document.addEventListener('DOMContentLoaded', function() {
    // Initialize any custom charts if needed
    console.log('Charts initialized');
    
    // Responsive chart resizing
    window.addEventListener('resize', function() {
        const charts = document.querySelectorAll('.plotly-graph-div');
        charts.forEach(chart => {
            Plotly.Plots.resize(chart);
        });
    });
    
    // Add loading states for charts
    const chartContainers = document.querySelectorAll('.chart-container');
    chartContainers.forEach(container => {
        container.innerHTML = '<div class="chart-loading">Loading chart...</div>';
    });
});