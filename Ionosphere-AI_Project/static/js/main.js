document.addEventListener("DOMContentLoaded", () => {

    // Small page entrance effect
    document.body.classList.add("page-ready");

    // Prevent accidental double-click navigation
    document.querySelectorAll("a").forEach(link => {

        link.addEventListener("click", () => {
            link.classList.add("loading-link");
        });

    });

});