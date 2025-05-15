import movies from './movieList.js';

document.addEventListener('DOMContentLoaded', function() {
    const input = document.getElementById('movie-search-input');
    const list = document.getElementById('autocomplete-list');

    input.addEventListener('input', function() {
        const val = this.value.trim().toLowerCase();
        list.innerHTML = '';
        if (!val) return;
        const filtered = movies.filter(movie => movie.toLowerCase().includes(val));
        filtered.forEach(movie => {
            const item = document.createElement('button');
            item.type = 'button';
            item.className = 'list-group-item list-group-item-action';
            item.textContent = movie;
            item.onclick = function() {
                input.value = movie;
                list.innerHTML = '';
            };
            list.appendChild(item);
        });
    });

    document.addEventListener('click', function(e) {
        if (!list.contains(e.target) && e.target !== input) {
            list.innerHTML = '';
        }
    });
});

async function getRecommendation() {
    const resultsDiv = document.getElementById("results-container");
    resultsDiv.innerHTML = ''; 

    const subheadingText = "You may also like the following movies:";
    resultsDiv.appendChild(document.createElement('h4')).textContent = subheadingText;

    const input = document.getElementById('movie-search-input');
    const selectedMovie = input.value.trim();
    if (!selectedMovie) return;

        try {
            const response = await fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ title: selectedMovie }) // Must match Flask
            });

            const data = await response.json();

            if (response.ok) {
        
                data.recommendations.forEach(item => {
                    console.log('Item:', item); 
                    const listItem = document.createElement('li');  // Create a new <li> element
                    listItem.textContent = item;  // Set the text content to the item
                    resultsDiv.appendChild(listItem);
                });
            } 
    } catch (error) {
        console.error('Fetch error:', error);
    }
}


// Attach to button
document.getElementById('search-button').addEventListener("click", function(e) {
    e.preventDefault(); // Prevents page reload
    getRecommendation();
});