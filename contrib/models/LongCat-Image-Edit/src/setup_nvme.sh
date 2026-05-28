#!/bin/bash
set -e

MOUNT_POINT="/opt/dlami/nvme"
RAID_DEVICE="/dev/md0"

echo "=== NVMe RAID0 Setup Script for trn2.48xlarge ==="

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)"
   exit 1
fi

# Check if already mounted
if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
    echo "$MOUNT_POINT is already mounted."
    df -h "$MOUNT_POINT"
    exit 0
fi

# Create mount point
mkdir -p "$MOUNT_POINT"

# Case 1: RAID device exists - just mount it
if [[ -e "$RAID_DEVICE" ]]; then
    echo "RAID device $RAID_DEVICE exists. Mounting..."
    mount "$RAID_DEVICE" "$MOUNT_POINT"
    chown ubuntu:ubuntu "$MOUNT_POINT"
    chmod 755 "$MOUNT_POINT"
    echo ""
    echo "=== Mount Complete ==="
    df -h "$MOUNT_POINT"
    exit 0
fi

# Case 2: Try to assemble from existing superblocks
echo "RAID device $RAID_DEVICE not found. Trying to assemble existing array..."
if mdadm --assemble --scan 2>/dev/null; then
    sleep 1
    if [[ -e "$RAID_DEVICE" ]]; then
        echo "RAID array reassembled successfully. Mounting..."
        mount "$RAID_DEVICE" "$MOUNT_POINT"
        chown ubuntu:ubuntu "$MOUNT_POINT"
        chmod 755 "$MOUNT_POINT"
        echo ""
        echo "=== Mount Complete ==="
        df -h "$MOUNT_POINT"
        exit 0
    fi
fi

# Case 3: Create new RAID
echo ""
echo "WARNING: No existing RAID array found."
echo "Creating a new RAID array will FORMAT and ERASE all data on NVMe devices!"
echo ""
read -p "Do you want to create a NEW RAID array? (yes/no): " CONFIRM

if [[ "$CONFIRM" != "yes" ]]; then
    echo "Aborted. No changes made."
    exit 1
fi

# Find root device and exclude it
ROOT_NVME=$(lsblk -n -o PKNAME,MOUNTPOINT | awk '$2=="/" {print $1; exit}')
echo "Root device detected: /dev/$ROOT_NVME (will be excluded)"

# Find all NVMe devices (excluding root)
NVME_DEVICES=$(lsblk -d -n -o NAME,TYPE | grep nvme | grep disk | awk '{print "/dev/"$1}' | grep -v "$ROOT_NVME" || true)
NVME_COUNT=$(echo "$NVME_DEVICES" | wc -l)

echo "Found $NVME_COUNT NVMe devices:"
echo "$NVME_DEVICES"

if [[ $NVME_COUNT -lt 1 ]]; then
    echo "No additional NVMe devices found."
    exit 1
fi

echo "Creating RAID0 array with $NVME_COUNT devices..."

for dev in $NVME_DEVICES; do
    mdadm --zero-superblock "$dev" 2>/dev/null || true
done

mdadm --create "$RAID_DEVICE" \
    --level=0 \
    --raid-devices=$NVME_COUNT \
    $NVME_DEVICES

echo "Formatting $RAID_DEVICE with ext4..."
mkfs.ext4 -F "$RAID_DEVICE"

echo "Mounting $RAID_DEVICE to $MOUNT_POINT..."
mount "$RAID_DEVICE" "$MOUNT_POINT"

chown ubuntu:ubuntu "$MOUNT_POINT"
chmod 755 "$MOUNT_POINT"

echo ""
echo "=== Setup Complete (New RAID Created) ==="
df -h "$MOUNT_POINT"
echo ""
echo "NVMe storage is now available at $MOUNT_POINT"
