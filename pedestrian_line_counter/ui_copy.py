from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UiPageCopy:
    title: str
    subtitle: str


LOGIN_PAGE = UiPageCopy(
    title="Operator Login",
    subtitle="Masuk untuk membuka dashboard, antrian review, dan detail event.",
)

DASHBOARD_PAGE = UiPageCopy(
    title="Traffic Monitoring Dashboard",
    subtitle="Total data dari live ditambah dengan data yang akan direview.",
)

REVIEW_PAGE = UiPageCopy(
    title="Antrian Review",
    subtitle="Periksa data kendaraan, lalu tandai sebagai diterima atau ditolak.",
)

EVENT_DETAIL_PAGE = UiPageCopy(
    title="Event Detail",
    subtitle="Bukti, metadata, dan hasil review untuk satu data.",
)
