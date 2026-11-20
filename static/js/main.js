/**
 * main.js
 * Core frontend script for sidebar toggle, live clock, modals, and toasts.
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Live Clock
    const clockEl = document.getElementById('liveClock');
    function updateClock() {
        if (!clockEl) return;
        const now = new Date();
        const timeStr = now.toLocaleTimeString('en-IN', { hour12: false });
        clockEl.textContent = `${timeStr} IST`;
    }
    updateClock();
    setInterval(updateClock, 1000);

    // 2. Sidebar Toggle (Mobile & Desktop)
    const toggleBtn = document.getElementById('toggleSidebar');
    const sidebar = document.getElementById('sidebar');
    const mainWrapper = document.getElementById('mainWrapper');

    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener('click', () => {
            if (window.innerWidth <= 992) {
                sidebar.classList.toggle('show');
            } else {
                sidebar.classList.toggle('collapsed');
                if (sidebar.classList.contains('collapsed')) {
                    sidebar.style.width = '78px';
                    mainWrapper.style.marginLeft = '78px';
                } else {
                    sidebar.style.width = '260px';
                    mainWrapper.style.marginLeft = '260px';
                }
            }
        });
    }

    // Close mobile sidebar on outer click
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 992 && sidebar && sidebar.classList.contains('show')) {
            if (!sidebar.contains(e.target) && !toggleBtn.contains(e.target)) {
                sidebar.classList.remove('show');
            }
        }
    });
});

/**
 * Opens the high-resolution image modal viewer.
 */
function viewImageModal(src, title = 'Inspection View', subtitle = '') {
    const modalImg = document.getElementById('modalImageSrc');
    const modalTitle = document.getElementById('modalImageTitle');
    const modalDetails = document.getElementById('modalImageDetails');
    const modalEl = document.getElementById('imageModal');

    if (modalImg && modalEl) {
        modalImg.src = src;
        if (modalTitle) modalTitle.innerHTML = `<i class="fa-solid fa-eye me-2 text-info"></i>${title}`;
        if (modalDetails) modalDetails.innerHTML = subtitle;
        const modal = new bootstrap.Modal(modalEl);
        modal.show();
    }
}
