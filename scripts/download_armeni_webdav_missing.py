#!/usr/bin/env python3
"""Download missing Armeni CTF .ds files from the Radboud WebDAV endpoint."""

from __future__ import annotations

import argparse
import getpass
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_WEBDAV_ROOT = "https://webdav.data.ru.nl/dccn/DSC_3011085.05_995_v1"
DAV_NS = "{DAV:}"
BLOCK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class DavEntry:
    href: str
    is_collection: bool
    size: int | None


@dataclass(frozen=True)
class TargetRecording:
    subject: str
    session: str
    task: str
    local_ds: Path
    remote_ds_url: str


class WebDavClient:
    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        timeout: float = 60.0,
        auth_root: str = DEFAULT_WEBDAV_ROOT,
    ):
        self.timeout = timeout
        self.opener = urllib.request.build_opener()
        if username:
            password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
            password_mgr.add_password(None, auth_root, username, password or "")
            self.opener.add_handler(urllib.request.HTTPBasicAuthHandler(password_mgr))

    def propfind(self, url: str, depth: int = 1) -> list[DavEntry]:
        req = urllib.request.Request(
            url,
            method="PROPFIND",
            headers={
                "Depth": str(depth),
                "Content-Type": "application/xml",
                "User-Agent": "ScraBrain-Armeni-WebDAV/1.0",
            },
            data=b"""<?xml version="1.0" encoding="utf-8" ?>
<propfind xmlns="DAV:">
  <prop>
    <resourcetype/>
    <getcontentlength/>
  </prop>
</propfind>
""",
        )
        with self.opener.open(req, timeout=self.timeout) as response:
            body = response.read()

        root = ET.fromstring(body)
        entries: list[DavEntry] = []
        for response in root.findall(f"{DAV_NS}response"):
            href_node = response.find(f"{DAV_NS}href")
            prop_node = response.find(f"{DAV_NS}propstat/{DAV_NS}prop")
            if href_node is None or prop_node is None or not href_node.text:
                continue

            resource_type = prop_node.find(f"{DAV_NS}resourcetype")
            is_collection = (
                resource_type is not None
                and resource_type.find(f"{DAV_NS}collection") is not None
            )
            size_node = prop_node.find(f"{DAV_NS}getcontentlength")
            size = int(size_node.text) if size_node is not None and size_node.text else None
            entries.append(DavEntry(href=href_node.text, is_collection=is_collection, size=size))
        return entries

    def download(self, url: str, local_path: Path, size: int | None, overwrite: bool) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)

        existing_size = local_path.stat().st_size if local_path.exists() else 0
        if size is not None and existing_size == size and not overwrite:
            print(f"skip    {local_path} ({format_bytes(size)})", flush=True)
            return

        headers = {}
        mode = "wb"
        if existing_size > 0 and size is not None and existing_size < size and not overwrite:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"
            print(
                f"resume  {local_path} "
                f"({format_bytes(existing_size)} / {format_bytes(size)})",
                flush=True,
            )
        elif existing_size > 0:
            print(f"replace {local_path}", flush=True)
        else:
            print(
                f"get     {local_path} ({format_bytes(size) if size is not None else 'unknown'})",
                flush=True,
            )

        headers["User-Agent"] = "ScraBrain-Armeni-WebDAV/1.0"
        req = urllib.request.Request(url, headers=headers)
        with self.opener.open(req, timeout=self.timeout) as response:
            if mode == "ab" and response.getcode() != 206:
                mode = "wb"
                existing_size = 0

            downloaded = existing_size
            last_report = time.monotonic()
            with local_path.open(mode + "") as handle:
                while True:
                    block = response.read(BLOCK_SIZE)
                    if not block:
                        break
                    handle.write(block)
                    downloaded += len(block)
                    now = time.monotonic()
                    if size is not None and now - last_report >= 10:
                        pct = 100.0 * downloaded / size
                        print(
                            f"        {pct:5.1f}% {format_bytes(downloaded)} / {format_bytes(size)}",
                            flush=True,
                        )
                        last_report = now


def format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{value} B"


def split_csv(values: str | None) -> list[str] | None:
    if values is None:
        return None
    return [value.strip() for value in values.split(",") if value.strip()]


def discover_targets_from_sidecars(
    dataset_root: Path,
    webdav_root: str,
    tasks: set[str],
) -> list[TargetRecording]:
    targets: list[TargetRecording] = []
    for sidecar in sorted(dataset_root.glob("sub-*/ses-*/meg/*_task-*_meg.json")):
        name = sidecar.name
        if "_task-emptyroom_" in name:
            continue

        parts = name.split("_")
        subject = parts[0]
        session = parts[1]
        task_part = next((part for part in parts if part.startswith("task-")), None)
        if task_part is None:
            continue
        task = task_part.removeprefix("task-")
        if task not in tasks:
            continue

        ds_name = name.removesuffix(".json") + ".ds"
        local_ds = sidecar.parent / ds_name
        remote_path = f"{subject}/{session}/meg/{ds_name}/"
        targets.append(
            TargetRecording(
                subject=subject,
                session=session,
                task=task,
                local_ds=local_ds,
                remote_ds_url=join_url(webdav_root, remote_path),
            )
        )
    return targets


def build_targets_from_args(
    dataset_root: Path,
    webdav_root: str,
    subjects: Iterable[str],
    sessions: Iterable[str],
    tasks: Iterable[str],
) -> list[TargetRecording]:
    targets: list[TargetRecording] = []
    for subject in subjects:
        for session in sessions:
            for task in tasks:
                ds_name = f"{subject}_{session}_task-{task}_meg.ds"
                targets.append(
                    TargetRecording(
                        subject=subject,
                        session=session,
                        task=task,
                        local_ds=dataset_root / subject / session / "meg" / ds_name,
                        remote_ds_url=join_url(webdav_root, f"{subject}/{session}/meg/{ds_name}/"),
                    )
                )
    return targets


def join_url(root: str, relative_path: str) -> str:
    return urllib.parse.urljoin(root.rstrip("/") + "/", relative_path)


def remote_file_url(webdav_root: str, href: str) -> str:
    parsed_root = urllib.parse.urlparse(webdav_root)
    return urllib.parse.urlunparse(
        (
            parsed_root.scheme,
            parsed_root.netloc,
            href,
            "",
            "",
            "",
        )
    )


def relative_file_name(ds_url: str, href: str) -> str:
    ds_path = urllib.parse.urlparse(ds_url).path
    if not ds_path.endswith("/"):
        ds_path += "/"
    if not href.startswith(ds_path):
        raise ValueError(f"Unexpected WebDAV href outside .ds directory: {href}")
    return urllib.parse.unquote(href[len(ds_path):])


def resolve_credentials(args: argparse.Namespace) -> tuple[str | None, str | None]:
    username = first_non_empty(args.username, os.environ.get("RDR_USERNAME"), os.environ.get("WEBDAV_USERNAME"))
    password = first_non_empty(args.password, os.environ.get("RDR_PASSWORD"), os.environ.get("WEBDAV_PASSWORD"))
    if username and password is None:
        password = getpass.getpass(f"Password for {username}: ")
    return username, password


def first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value:
            return value
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download missing CTF files for Armeni task-compr recordings via WebDAV. "
            "By default it uses local BIDS sidecars as the manifest."
        )
    )
    parser.add_argument("--dataset-root", default="datasets/armeni", type=Path)
    parser.add_argument("--webdav-root", default=DEFAULT_WEBDAV_ROOT)
    parser.add_argument("--tasks", default="compr", help="Comma-separated task names.")
    parser.add_argument("--subjects", help="Comma-separated subjects, e.g. sub-001,sub-002.")
    parser.add_argument("--sessions", help="Comma-separated sessions, e.g. ses-001,ses-002.")
    parser.add_argument("--username", help="Optional WebDAV username.")
    parser.add_argument("--password", help="Optional WebDAV password.")
    parser.add_argument("--dry-run", action="store_true", help="List files that would be downloaded.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing files even if sizes differ.")
    parser.add_argument("--timeout", default=60.0, type=float, help="Network timeout in seconds.")
    parser.add_argument(
        "--required-only",
        action="store_true",
        default=True,
        help="Download only .res4 and .meg4 files. This is the default.",
    )
    parser.add_argument(
        "--all-files",
        action="store_false",
        dest="required_only",
        help="Download every file in each .ds directory instead of only .res4 and .meg4.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root
    tasks = set(split_csv(args.tasks) or ["compr"])

    subjects = split_csv(args.subjects)
    sessions = split_csv(args.sessions)
    if subjects or sessions:
        if not subjects or not sessions:
            print("--subjects and --sessions must be provided together.", file=sys.stderr)
            return 2
        targets = build_targets_from_args(dataset_root, args.webdav_root, subjects, sessions, tasks)
    else:
        targets = discover_targets_from_sidecars(dataset_root, args.webdav_root, tasks)

    if not targets:
        print(
            f"No local task sidecars found under {dataset_root}; nothing to download.",
            file=sys.stderr,
        )
        return 1

    username, password = resolve_credentials(args)
    client = WebDavClient(
        username=username,
        password=password,
        timeout=args.timeout,
        auth_root=args.webdav_root,
    )

    total_files = 0
    total_bytes = 0
    for target in targets:
        try:
            entries = client.propfind(target.remote_ds_url, depth=1)
        except urllib.error.HTTPError as exc:
            print(f"missing remote {target.remote_ds_url}: HTTP {exc.code}", file=sys.stderr)
            continue

        for entry in entries:
            if entry.is_collection:
                continue
            file_name = relative_file_name(target.remote_ds_url, entry.href)
            if args.required_only and not (file_name.endswith(".res4") or file_name.endswith(".meg4")):
                continue

            local_path = target.local_ds / file_name
            if local_path.exists() and entry.size is not None and local_path.stat().st_size == entry.size:
                continue

            total_files += 1
            if entry.size is not None:
                current_size = local_path.stat().st_size if local_path.exists() else 0
                total_bytes += max(entry.size - current_size, 0)

            url = remote_file_url(args.webdav_root, entry.href)
            if args.dry_run:
                print(f"would get {local_path} ({format_bytes(entry.size)})", flush=True)
            else:
                try:
                    client.download(url, local_path, entry.size, overwrite=args.overwrite)
                except urllib.error.HTTPError as exc:
                    if exc.code == 401:
                        print(
                            "\nWebDAV returned HTTP 401 while downloading a file. "
                            "The repository is listing metadata but requires credentials for file downloads.\n"
                            "Relaunch with credentials, for example:\n"
                            "  RDR_USERNAME='<user>' RDR_PASSWORD='<password>' "
                            "bash run_armeni_webdav_download.sh --replace\n",
                            file=sys.stderr,
                        )
                        return 1
                    raise

    if args.dry_run:
        print(f"Dry run: {total_files} file(s), about {format_bytes(total_bytes)} missing.", flush=True)
    else:
        print(
            f"Done: checked {len(targets)} recording(s), downloaded/updated {total_files} file(s).",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
